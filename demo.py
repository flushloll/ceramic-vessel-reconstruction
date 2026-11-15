"""Make demo/demo.gif on a synthetic pot (no real scan data needed).

1. Build a wheel-thrown-style jar (a surface of revolution) and tilt it randomly.
2. Keep only a broken-off fragment of its surface.
3. Estimate the jar's symmetry axis from the fragment alone (symhybrid_core.mesh_axis).
4. Revolve the fragment around that axis to rebuild the wall at the heights the fragment covers.

In the full method a learned point-cloud model fills the heights the fragment never reaches; that model and
its training data are not in this repo, so the demo shows the geometric part only.

    python demo.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from symhybrid_core import fps, mesh_axis, revolve

OUT = "demo"
rng = np.random.default_rng(4)


def jar(nh=60, na=96):
    """Vertices/faces of a jar profile r(h) revolved about +Y."""
    h = np.linspace(0, 1.6, nh)
    r = 0.28 + 0.5 * np.sin(np.pi * h / 1.75) ** 1.3 + 0.14 * np.clip((h - 1.4) / 0.2, 0, 1) ** 2
    a = np.linspace(0, 2 * np.pi, na, endpoint=False)
    V = np.stack([np.outer(r, np.cos(a)), np.repeat(h[:, None], na, 1), np.outer(r, np.sin(a))], -1).reshape(-1, 3)
    F = []
    for j in range(nh - 1):
        for k in range(na):
            p, q = j * na + k, j * na + (k + 1) % na
            F += [[p, p + na, q], [q, p + na, q + na]]
    return V, np.asarray(F), h, a


def random_rotation():
    q = rng.normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main():
    os.makedirs(OUT, exist_ok=True)
    V, F, _, _ = jar()
    R = random_rotation(); Vr = V @ R.T
    true_axis = R @ np.array([0.0, 1.0, 0.0])

    # fragment: faces inside an irregular wedge, covering only part of the jar's height
    cent = V[F].mean(1)
    ang = np.arctan2(cent[:, 2], cent[:, 0])
    edge = 0.12 * np.sin(7 * cent[:, 1]) + 0.08 * np.sin(13 * cent[:, 1] + 1)
    keep = (ang > -0.45 + edge) & (ang < 0.95 + edge) & (cent[:, 1] > 0.35 + 0.05 * np.sin(5 * ang)) \
        & (cent[:, 1] < 1.2 + 0.06 * np.sin(4 * ang))
    frag = Vr[F[keep]]
    # (clean surface; real scans are smoothed first, see intake.taubin_smooth)

    d, c = mesh_axis(frag)
    err = np.degrees(np.arccos(min(1.0, abs(d @ true_axis))))
    print(f"fragment: {keep.sum()} of {len(F)} faces ({keep.mean():.0%} of the surface); axis error {err:.2f} deg")

    frag_pts = fps(frag.reshape(-1, 3), 500)
    recon = fps(revolve(frag_pts, c, d, K=48), 2500)
    full = fps(Vr, 2500)
    tr = (full - c) @ d
    axis_line = c + np.array([tr.min() - 0.15, tr.max() + 0.15])[:, None] * d
    # the fit above ran on the tilted pot; draw everything turned back upright so it reads as a pot
    up = lambda p: p @ R
    frag_pts, recon, full, axis_line = up(frag_pts), up(recon), up(full), up(axis_line)
    swap = lambda p: p[:, [0, 2, 1]]                          # matplotlib's vertical axis is z
    frag_pts, recon, full, axis_line = swap(frag_pts), swap(recon), swap(full), swap(axis_line)

    fig = plt.figure(figsize=(5.2, 5.2))
    ax = fig.add_subplot(projection="3d")
    fig.subplots_adjust(0, 0, 1, 0.9)
    allp = np.concatenate([full, recon])
    lo, hi = allp.min(0), allp.max(0); mid = (lo + hi) / 2; s = (hi - lo).max() / 2 * 0.8
    n = 72

    def draw(i):
        ax.cla(); ax.set_axis_off()
        ax.set_xlim(mid[0] - s, mid[0] + s); ax.set_ylim(mid[1] - s, mid[1] + s); ax.set_zlim(mid[2] - s, mid[2] + s)
        ax.view_init(elev=12, azim=i * 360 / n)
        ax.scatter(*full.T, s=2, c="#9a9a9a", alpha=0.35)
        if i >= n // 3:
            ax.scatter(*recon.T, s=2.5, c="#3b82c4", alpha=0.8)
            ax.plot(*axis_line.T, color="#222222", lw=1.5)
        ax.scatter(*frag_pts.T, s=6, c="#c0563a")
        ax.set_title("fragment → estimated axis → revolved wall" if i >= n // 3 else "a broken fragment (red) of a pot (grey)",
                     fontsize=10)
        return []

    caption = fig.text(0.5, 0.03, "", ha="center", fontsize=9)

    def step(i):
        draw(i)
        caption.set_text(f"axis error {err:.1f}° · blue = rebuilt from the fragment · grey = still missing"
                         if i >= n // 3 else "")
        return []

    FuncAnimation(fig, step, frames=n).save(os.path.join(OUT, "demo.gif"), writer=PillowWriter(fps=12), dpi=70)
    print("wrote", os.path.join(OUT, "demo.gif"))


if __name__ == "__main__":
    main()
