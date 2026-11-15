"""SymHybrid v2 core (2026-11-04). NumPy only. Observation-only.

Identical composition to core.py (v1) except the rotational axis: v2 estimates it from the exact
area-weighted face normals of the surviving fragment MESH (trimmed weighted linear line complex),
instead of PCA normals of the 384 sampled points. Evidence: plan section 18 / 19.

predict(X, deployed_points, triangles):
  X               [384,3] normalized observation points (same as deployed308's input)
  deployed_points [384,3] deployed308's normalized whole-surface estimate
  triangles       [F,3,3] surviving fragment triangles in the same normalized frame
"""
import numpy as np

MAX_FACES = 30000


def fps(pts, k, seed=0):
    n = len(pts); idx = np.empty(k, int); idx[0] = np.random.default_rng(seed).integers(n)
    d = np.full(n, np.inf)
    for i in range(1, k):
        d = np.minimum(d, ((pts - pts[idx[i - 1]]) ** 2).sum(1)); idx[i] = int(d.argmax())
    return pts[idx]


def axis_frame(d):
    d = d / np.linalg.norm(d); a = np.array([1.0, 0, 0]) if abs(d[0]) < .9 else np.array([0, 0, 1.0])
    u = np.cross(d, a); u /= np.linalg.norm(u); w = np.cross(d, u); return d, u, w


def revolve(points, c, d, K=32):
    d, u, w = axis_frame(d); q = points - c
    h = q @ d; x = q @ u; y = q @ w
    out = []
    for t in np.linspace(0, 2 * np.pi, K, endpoint=False):
        ct, st = np.cos(t), np.sin(t)
        out.append(c + np.outer(h, d) + np.outer(x * ct - y * st, u) + np.outer(x * st + y * ct, w))
    return np.concatenate(out)


def mesh_axis(triangles, trim=.8, iters=3):
    """Axis = least-squares line meeting every face-normal line (Pluecker coordinates), area-weighted, trimmed."""
    tri = np.asarray(triangles, float)
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); area = np.linalg.norm(cr, axis=1) / 2; ok = area > 0
    tri, cr, area = tri[ok], cr[ok], area[ok]
    if len(tri) < 3: raise ValueError('symhybrid_v2_too_few_faces')
    nrm = cr / (2 * area[:, None]); cent = tri.mean(1)
    if len(tri) > MAX_FACES:
        k = np.random.default_rng(0).choice(len(tri), MAX_FACES, replace=False); cent, nrm, area = cent[k], nrm[k], area[k]
    m0 = (cent * area[:, None]).sum(0) / area.sum(); s0 = np.linalg.norm(cent - m0, axis=1).max(); p = (cent - m0) / s0
    L = np.hstack([np.cross(p, nrm), nrm]); w = area / area.sum()
    for _ in range(iters):
        A = (L * w[:, None]).T @ L; A11, A12, A22 = A[:3, :3], A[:3, 3:], A[3:, 3:]
        S = A11 - A12 @ np.linalg.pinv(A22) @ A12.T; _, V = np.linalg.eigh(S); c = V[:, 0]; cb = -np.linalg.pinv(A22) @ A12.T @ c
        r = (L @ np.concatenate([c, cb])) ** 2; keep = r <= np.quantile(r, trim); w = np.where(keep, area, 0.0); w = w / w.sum()
    d = c / np.linalg.norm(c); return d, m0 + s0 * np.cross(d, cb)


def predict(X, deployed_points, triangles):
    X = np.asarray(X, float); dp = np.asarray(deployed_points, float)
    if X.shape != (384, 3) or dp.shape != (384, 3) or not (np.isfinite(X).all() and np.isfinite(dp).all()):
        raise ValueError('symhybrid_input_contract')
    d, c = mesh_axis(triangles)
    hx = (X - c) @ d; lo, hi = hx.min(), hx.max(); hd = (dp - c) @ d
    outb = dp[(hd < lo) | (hd > hi)]; n_out = len(outb)
    parts = []
    if 384 - n_out > 0: parts.append(fps(revolve(X, c, d), 384 - n_out))
    if n_out: parts.append(fps(revolve(outb, c, d), n_out))
    P = np.concatenate(parts)
    if P.shape != (384, 3) or not np.isfinite(P).all(): raise ValueError('symhybrid_output_contract')
    return P
