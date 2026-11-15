"""Real-sherd intake: scanned mesh -> surviving-surface observation the completion models accept.

NumPy only. Steps (see 2026-11-04-real-fragment-intake-plan.md):
  load (OBJ / PLY / STL) -> weld -> keep large connected pieces -> decimate (vertex clustering)
  -> segment walls vs break faces -> orient (axis upright) -> observation dict + confidence report.

Wall/break segmentation uses two facts about wheel-thrown sherds: (1) on a surface of revolution every
face-normal line passes through the symmetry axis, break faces' normal lines do not; (2) break faces meet
the walls at sharp creases. Faces are grouped into smooth patches at creases, the axis is fitted robustly
to all faces, and each patch is labelled wall if most of its area has normal lines passing near the axis.
"""
from __future__ import annotations
import struct
import numpy as np

# ----------------------------------------------------------------------------------------------- loading

def load_mesh(path: str) -> tuple[np.ndarray, np.ndarray]:
    p = str(path).lower()
    if p.endswith('.obj'): return _load_obj(path)
    if p.endswith('.ply'): return _load_ply(path)
    if p.endswith('.stl'): return _load_stl(path)
    raise ValueError('unsupported mesh format (use OBJ, PLY or STL): ' + path)


def _load_obj(path):
    vs, fs = [], []
    with open(path, 'rb') as fh:
        for raw in fh:
            if raw.startswith(b'v '):
                vs.append(raw[2:].split()[:3])
            elif raw.startswith(b'f '):
                idx = [int(t.split(b'/')[0]) for t in raw[2:].split()]
                fs.extend([idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1))   # fan triangulation
    V = np.asarray(vs, dtype=np.float64); F = np.asarray(fs, dtype=np.int64)
    if len(F) == 0: raise ValueError('OBJ has no faces')
    F = np.where(F < 0, F + len(V), F - 1)                                                # 1-based / negative indices
    return V, F


def _load_stl(path):
    raw = open(path, 'rb').read()
    if raw[:5].lower() == b'solid' and b'facet' in raw[:1000]:
        pts = [l.split()[1:4] for l in raw.decode('ascii', 'ignore').splitlines() if l.strip().startswith('vertex')]
        V = np.asarray(pts, float)
    else:
        n = struct.unpack('<I', raw[80:84])[0]
        rec = np.frombuffer(raw, dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]), count=n, offset=84)
        V = rec['v'].reshape(-1, 3).astype(float)
    return V, np.arange(len(V)).reshape(-1, 3)


def _load_ply(path):
    raw = open(path, 'rb').read(); end = raw.index(b'end_header') + len(b'end_header'); end = raw.index(b'\n', end) + 1
    head = raw[:end].decode('ascii', 'ignore').splitlines(); fmt = next(l.split()[1] for l in head if l.startswith('format'))
    T = {'char': 'i1', 'uchar': 'u1', 'int8': 'i1', 'uint8': 'u1', 'short': 'i2', 'ushort': 'u2', 'int16': 'i2', 'uint16': 'u2',
         'int': 'i4', 'uint': 'u4', 'int32': 'i4', 'uint32': 'u4', 'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8'}
    elems, cur = [], None
    for l in head:
        t = l.split()
        if t[:1] == ['element']: cur = [t[1], int(t[2]), []]; elems.append(cur)
        elif t[:1] == ['property']: cur[2].append(t[1:])
    body = raw[end:]
    if fmt == 'ascii':
        lines = body.decode('ascii', 'ignore').split('\n'); pos = 0; V = F = None
        for name, n, props in elems:
            rows = lines[pos:pos + n]; pos += n
            if name == 'vertex':
                names = [p[-1] for p in props]; ix = [names.index(k) for k in 'xyz']
                V = np.asarray([[float(r.split()[i]) for i in ix] for r in rows])
            elif name == 'face':
                fs = []
                for r in rows:
                    t = [int(x) for x in r.split()]; k = t[0]; idx = t[1:1 + k]
                    fs.extend([idx[0], idx[i], idx[i + 1]] for i in range(1, k - 1))
                F = np.asarray(fs, np.int64)
        return V, F
    end_ch = '<' if 'little' in fmt else '>'; off = 0; V = F = None
    for name, n, props in elems:
        if all(p[0] != 'list' for p in props):
            dt = np.dtype([(p[1], end_ch + T[p[0]]) for p in props]); arr = np.frombuffer(body, dt, n, off); off += dt.itemsize * n
            if name == 'vertex': V = np.stack([arr[k].astype(float) for k in 'xyz'], 1)
        else:                                                                                  # face list (assume triangles fast path)
            cnt_t, idx_t = end_ch + T[props[0][1]], end_ch + T[props[0][2]]
            c0 = np.frombuffer(body, cnt_t, 1, off)[0]
            dt = np.dtype([('c', cnt_t), ('i', idx_t, (int(c0),))])
            arr = np.frombuffer(body, dt, n, off)
            if not np.all(arr['c'] == c0): raise ValueError('mixed polygon sizes in binary PLY not supported')
            off += dt.itemsize * n; idx = arr['i'].astype(np.int64)
            F = np.concatenate([idx[:, [0, i, i + 1]] for i in range(1, idx.shape[1] - 1)]) if name == 'face' else F
    return V, F

# ----------------------------------------------------------------------------------------------- cleaning

def weld(V, F, rel_tol=1e-7):
    diag = np.linalg.norm(np.ptp(V, 0)) or 1.0; q = np.round(V / (diag * rel_tol)).astype(np.int64)
    _, keep, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    F2 = inv.reshape(-1)[F]; good = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
    return V[keep], F2[good]


def face_geometry(V, F):
    t = V[F]; cr = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]); a = np.linalg.norm(cr, axis=1) / 2
    n = np.divide(cr, 2 * a[:, None], out=np.zeros_like(cr), where=a[:, None] > 0)
    return t.mean(1), n, a


def _union_find(n, pairs):
    """Connected-component labels (vectorized min-label propagation with pointer jumping)."""
    lab = np.arange(n)
    pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
    if len(pairs) == 0: return lab
    a, b = pairs[:, 0], pairs[:, 1]
    while True:
        m = np.minimum(lab[a], lab[b]); new = lab.copy(); np.minimum.at(new, a, m); np.minimum.at(new, b, m)
        new = new[new]                                                      # pointer jumping
        if np.array_equal(new, lab): return lab
        lab = new


def _union_find_reference(n, pairs):
    parent = np.arange(n)
    def find(x):
        r = x
        while parent[r] != r: r = parent[r]
        while parent[x] != r: parent[x], x = r, parent[x]
        return r
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    return np.array([find(i) for i in range(n)])


def edge_adjacency(F):
    """Pairs of faces sharing an edge (manifold-ish meshes)."""
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1); fid = np.tile(np.arange(len(F)), 3)
    order = np.lexsort((e[:, 1], e[:, 0])); e, fid = e[order], fid[order]
    same = np.all(e[1:] == e[:-1], 1)
    return np.stack([fid[:-1][same], fid[1:][same]], 1), e


def largest_pieces(V, F, min_share=0.05):
    pairs, _ = edge_adjacency(F); lab = _union_find(len(F), pairs); _, _, a = face_geometry(V, F)
    tot = np.bincount(np.unique(lab, return_inverse=True)[1], weights=a)
    keep_lab = np.unique(lab)[tot >= min_share * tot.max()]
    m = np.isin(lab, keep_lab); return compact(V, F[m])


def compact(V, F):
    used, inv = np.unique(F, return_inverse=True); return V[used], inv.reshape(F.shape)


def decimate(V, F, max_vertices=90000):
    """Vertex clustering on a uniform grid sized to land under max_vertices; keeps the outline and walls."""
    if len(V) <= max_vertices: return V, F
    _, _, a = face_geometry(V, F); cell = np.sqrt(a.sum() / max_vertices) * 0.5   # surfaces: vertex count ~ area / cell^2
    for _ in range(60):
        key = np.floor((V - V.min(0)) / cell).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.reshape(-1)
        if inv.max() + 1 <= max_vertices: break
        cell *= 1.08
    W = np.zeros((inv.max() + 1, 3)); np.add.at(W, inv, V); W /= np.bincount(inv)[:, None]
    F2 = inv[F]; good = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2]); F2 = F2[good]
    _, first = np.unique(np.sort(F2, 1), axis=0, return_index=True)                  # drop duplicates, KEEP winding
    return compact(W, F2[np.sort(first)])

# ------------------------------------------------------------------------------------------------ axis

def cylinder_axis(cent, nrm, w, trim=.8, iters=4):
    """Degenerate case (near-cylindrical walls): all normals ~perpendicular to the axis. Direction = least-variance
    normal direction; point = least-squares intersection of the normal lines projected onto the cross-section."""
    ww = w / w.sum()
    for _ in range(iters):
        N = (nrm * ww[:, None]).T @ nrm; ev, E = np.linalg.eigh(N); d = E[:, 0]
        u = np.cross(d, [1.0, 0, 0] if abs(d[0]) < .9 else [0, 0, 1.0]); u /= np.linalg.norm(u); v = np.cross(d, u)
        P2 = np.stack([cent @ u, cent @ v], 1); n2 = np.stack([nrm @ u, nrm @ v], 1); n2 /= np.maximum(np.linalg.norm(n2, axis=1)[:, None], 1e-12)
        perp = np.stack([-n2[:, 1], n2[:, 0]], 1)                       # each 2-D normal line: perp . x = perp . p
        A = (perp * ww[:, None]).T @ perp; b = ((perp * (perp * P2).sum(1)[:, None]) * ww[:, None]).sum(0)
        x = np.linalg.lstsq(A, b, rcond=None)[0]; point = x[0] * u + x[1] * v
        r = normal_line_distance(cent, nrm, d, point); ww = np.where(r <= np.quantile(r, trim), w, 0.0); ww = ww / ww.sum()
    return d, point


def line_complex_axis(cent, nrm, w, trim=.8, iters=4, cyl_ratio=0.02):
    """Least-squares line meeting all face-normal lines (Pluecker), weighted, trimmed. Returns dir, point, per-face residual.
    Falls back to `cylinder_axis` when the normals are (nearly) all perpendicular to one direction, where the line-complex
    solution is degenerate (a whole family of lines fits)."""
    w = np.asarray(w, float)
    m0 = (cent * w[:, None]).sum(0) / w.sum(); s0 = np.linalg.norm(cent - m0, axis=1).max() or 1.0
    N = (nrm * (w / w.sum())[:, None]).T @ nrm; ev = np.linalg.eigvalsh(N)
    # robust version of the check: ignore the 20% of faces least consistent with the best plane of normals
    e0 = np.linalg.eigh(N)[1][:, 0]; dev = np.abs(nrm @ e0); core = dev <= np.quantile(dev, .8)
    Nc = (nrm[core] * (w[core] / w[core].sum())[:, None]).T @ nrm[core]; evc = np.linalg.eigvalsh(Nc)
    if evc[0] / max(evc[-1], 1e-30) < cyl_ratio:
        d, point = cylinder_axis(cent, nrm, w, trim, iters)
        return d, point, normal_line_distance(cent, nrm, d, point) / s0
    p = (cent - m0) / s0
    L = np.hstack([np.cross(p, nrm), nrm]); ww = w / w.sum()
    for _ in range(iters):
        A = (L * ww[:, None]).T @ L; A11, A12, A22 = A[:3, :3], A[:3, 3:], A[3:, 3:]
        S = A11 - A12 @ np.linalg.pinv(A22) @ A12.T; _, Vv = np.linalg.eigh(S); c = Vv[:, 0]; cb = -np.linalg.pinv(A22) @ A12.T @ c
        r = np.abs(L @ np.concatenate([c, cb])); ww = np.where(r <= np.quantile(r, trim), w, 0.0); ww = ww / ww.sum()
    d = c / np.linalg.norm(c); point = m0 + s0 * np.cross(d, cb)
    return d, point, normal_line_distance(cent, nrm, d, point) / s0


def normal_line_distance(cent, nrm, d, point):
    """Distance between each face-normal line and the axis line (0 for an ideal surface of revolution)."""
    cr = np.cross(nrm, d); nc = np.linalg.norm(cr, axis=1); w = point - cent
    par = nc < 1e-9
    dist = np.abs((w * cr).sum(1)) / np.where(par, 1, nc)
    dist[par] = np.linalg.norm(np.cross(w[par], d), axis=1)
    return dist

# ------------------------------------------------------------------------------------------- segmentation

def plate_normals(cent, area, radius, anchors=2500, seed=0):
    """Local plate normal at every face: smallest principal direction of nearby surface points (both walls).
    A sherd is a thin curved plate, so this is the wall direction even at rounded, eroded break edges."""
    rng = np.random.default_rng(seed); n = len(cent)
    A = cent[rng.choice(n, min(anchors, n), replace=False, p=area / area.sum())]
    sample = cent[rng.choice(n, min(40000, n), replace=False)]
    normals = np.empty((len(A), 3)); centers = np.empty((len(A), 3)); frames = np.empty((len(A), 3, 3)); coef = np.zeros((len(A), 6))
    for s in range(0, len(A), 256):
        d2 = ((A[s:s + 256, None] - sample[None]) ** 2).sum(-1)
        for k, row in enumerate(d2):
            q = sample[row <= radius * radius]
            if len(q) < 12: q = sample[np.argsort(row)[:48]]
            m = q.mean(0); E = np.linalg.eigh((q - m).T @ (q - m))[1]          # columns: plate normal, then in-plane axes
            uvw = (q - m) @ E[:, [2, 1, 0]]; u, v, w = uvw.T
            # quadratic mid-surface w = f(u, v): follows wall curvature, lies between the two wall layers
            coef[s + k] = np.linalg.lstsq(np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], 1), w, rcond=None)[0]
            normals[s + k] = E[:, 0]; centers[s + k] = m; frames[s + k] = E[:, [2, 1, 0]]
    owner = np.empty(n, np.int64)
    for s in range(0, n, 4096): owner[s:s + 4096] = ((cent[s:s + 4096, None] - A[None]) ** 2).sum(-1).argmin(1)
    uvw = np.einsum('nij,ni->nj', frames[owner], cent - centers[owner]); u, v, w = uvw.T; C = coef[owner]
    side = w - (C[:, 0] + C[:, 1] * u + C[:, 2] * v + C[:, 3] * u * u + C[:, 4] * u * v + C[:, 5] * v * v)   # along the anchor's plate normal
    return normals[owner], side


def _components(F, mask):
    pairs, _ = edge_adjacency(F); keep = mask[pairs[:, 0]] & mask[pairs[:, 1]]
    lab = _union_find(len(F), pairs[keep]); lab[~mask] = -1; return lab


def segment(V, F, wall_cos=0.7, radius_frac=0.15):
    """Label faces: 1 = outer wall, 2 = inner wall, 0 = break / other. Returns labels and axis info.
    Walls face along the local plate normal; break faces face sideways (|n . plate| small)."""
    cent, nrm, area = face_geometry(V, F); ok = area > 0
    ext = np.linalg.norm(np.ptp(cent[ok], 0)); radius = radius_frac * ext
    plate_ok, side_ok = plate_normals(cent[ok], area[ok], radius)
    plate = np.zeros_like(cent); side_raw = np.zeros(len(F)); plate[ok] = plate_ok; side_raw[ok] = side_ok
    align = np.zeros(len(F)); align[ok] = np.abs((nrm[ok] * plate[ok]).sum(1))
    wall = ok & (align >= wall_cos)
    labels = np.zeros(len(F), np.int8)
    if wall.sum() < 3:
        return labels, dict(axis_direction=np.array([0, 1.0, 0]), axis_point=cent.mean(0), wall_share=0.0, break_share=1.0, axis_residual_median=float('nan'), sheets=0)
    d, c, _ = line_complex_axis(cent[wall], nrm[wall], area[wall])
    # outer vs inner: each plate neighbourhood spans both walls, so its mean point lies mid-way through the wall.
    # Orient plate normals away from the axis; faces on the outer side of the mid-surface are outer wall.
    q = cent - c; radial = q - np.outer(q @ d, d); rhat = radial / np.maximum(np.linalg.norm(radial, axis=1)[:, None], 1e-12)
    side = side_raw * np.sign((plate * rhat).sum(1) + 1e-12)          # positive = away from the axis
    pos, neg = wall & (side > 0), wall & (side <= 0)
    balance = area[pos].sum() / max(area[wall].sum(), 1e-30); sep = np.median(np.abs(side[wall])) / radius
    two_sided = 0.2 <= balance <= 0.8 and sep >= 0.05
    if two_sided:
        labels[pos] = 1; labels[neg] = 2; sheets = 2
    else:
        labels[wall] = 1; sheets = 1                                   # single-sided surface (e.g. one face scanned)
    w2 = labels > 0
    d, c, res = line_complex_axis(cent[w2], nrm[w2], area[w2])       # final axis on all wall faces
    return labels, dict(axis_direction=d, axis_point=c, wall_share=float(area[w2].sum() / area[ok].sum()),
                        break_share=float(area[ok & ~w2].sum() / area[ok].sum()), axis_residual_median=float(np.median(res)), sheets=int(sheets), side_balance=float(balance), side_separation=float(sep))

# ----------------------------------------------------------------------------------------- orientation

def rotation_to(d, target=np.array([0.0, 1.0, 0.0])):
    d = d / np.linalg.norm(d); v = np.cross(d, target); s = np.linalg.norm(v); c = float(d @ target)
    if s < 1e-12: return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]); return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)

# ----------------------------------------------------------------------------------------------- driver

def taubin_smooth(V, F, iters=10, lam=0.5, mu=-0.53):
    """Shape-preserving denoising (Taubin lambda/mu): removes scan noise from normals without shrinking the sherd."""
    if iters <= 0: return V
    e = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0)
    deg = np.bincount(e.ravel(), minlength=len(V)).astype(float); deg[deg == 0] = 1; V = V.copy()
    for k in range(2 * iters):
        s = np.zeros_like(V); np.add.at(s, e[:, 0], V[e[:, 1]]); np.add.at(s, e[:, 1], V[e[:, 0]])
        V = V + (lam if k % 2 == 0 else mu) * (s / deg[:, None] - V)
    return V


AXIS_OK_DEG, AXIS_BAD_DEG = 10.0, 30.0       # verdict thresholds (see 2026-11-05 calibration in the intake plan)
RADIUS_OK_SPREAD = 0.25                      # sub-crop radius spread that counts as "as bad as" AXIS_OK_DEG
FLAT_RADIUS_OVER_SIZE = 6.0                  # fitted radius beyond this many sherd sizes: curvature too slight to place the axis


def _angle(a, b): return float(np.degrees(np.arccos(min(1.0, abs(float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))))))


def axis_checks(V, F, labels=None, crops=6, share=0.7, seed=0, max_faces=20000):
    """Ground-truth-free axis confidence. Sub-crop stability: re-fit the axis on `crops` random contiguous `share` crops
    of the wall and report the angle to the full-wall axis. Outer vs inner: when both walls are present, the angle
    between their independently fitted axes. Verdict: 'well determined' / 'uncertain' / 'undetermined'."""
    rng = np.random.default_rng(seed); c, n, a = face_geometry(V, F)
    wall = (a > 0) if labels is None else ((labels > 0) & (a > 0))
    outer = wall if labels is None or not (labels == 1).any() else ((labels == 1) & (a > 0))
    idx = np.flatnonzero(outer)
    if len(idx) > max_faces: idx = rng.choice(idx, max_faces, replace=False, p=a[idx] / a[idx].sum())
    d0, p0, res = line_complex_axis(c[idx], n[idx], a[idx]); angles = []; radii = []
    def mean_radius(d, p, m):
        q = c[m] - p; return float(np.average(np.linalg.norm(q - np.outer(q @ d, d), axis=1), weights=a[m]))
    size = float(np.linalg.norm(np.ptp(c[idx], 0))) or 1.0
    for _ in range(crops):
        seedp = c[rng.choice(idx, p=a[idx] / a[idx].sum())]; o = idx[np.argsort(((c[idx] - seedp) ** 2).sum(1))]
        keep = o[:np.searchsorted(np.cumsum(a[o]) / a[o].sum(), share) + 1]
        dk, pk, _ = line_complex_axis(c[keep], n[keep], a[keep]); angles.append(_angle(dk, d0)); radii.append(mean_radius(dk, pk, keep))
    r0 = mean_radius(d0, p0, idx); spread = float((np.max(radii) - np.min(radii)) / max(np.median(radii), 1e-12))
    out = dict(subcrop_deg_median=round(float(np.median(angles)), 2), subcrop_deg_max=round(float(np.max(angles)), 2),
               radius_over_size=round(r0 / size, 3), subcrop_radius_spread=round(spread, 3), axis_residual=round(float(np.median(res)), 4))
    worst = out['subcrop_deg_median']
    # the axis location (vessel radius) must also be stable: a flat sherd fixes a direction but not where the axis is
    worst = max(worst, AXIS_OK_DEG * spread / RADIUS_OK_SPREAD)
    if labels is not None and (labels == 1).sum() > 100 and (labels == 2).sum() > 100:
        m2 = (labels == 2) & (a > 0); out['outer_vs_inner_deg'] = round(_angle(d0, line_complex_axis(c[m2], n[m2], a[m2])[0]), 2); worst = max(worst, out['outer_vs_inner_deg'])
    if out['radius_over_size'] > FLAT_RADIUS_OVER_SIZE: worst = max(worst, AXIS_BAD_DEG + 1)
    out['verdict'] = 'well determined' if worst <= AXIS_OK_DEG else 'undetermined' if worst > AXIS_BAD_DEG else 'uncertain'
    return out


def intake(V, F, keep='outer', max_vertices=90000, canonicalize=True, flip=False, smooth_iters=10, checks=True):
    """Return (observation dict, report, R) where R rotates input coordinates into the observation frame."""
    V, F = weld(np.asarray(V, float), np.asarray(F, np.int64)); V, F = largest_pieces(V, F); V, F = decimate(V, F, max_vertices)
    V = taubin_smooth(V, F, smooth_iters)
    labels, info = segment(V, F)
    sel = labels == 1 if keep == 'outer' else labels > 0 if keep == 'walls' else np.ones(len(F), bool)
    if sel.sum() < 24: sel = np.ones(len(F), bool); info['fallback'] = 'segmentation_kept_too_few_faces'
    Vo, Fo = compact(V, F[sel])
    R = rotation_to(-info['axis_direction'] if flip else info['axis_direction']) if canonicalize else np.eye(3)
    Vo = Vo @ R.T
    report = dict(faces_in=int(len(F)), faces_kept=int(sel.sum()), wall_share=info['wall_share'], break_share=info['break_share'],
                  axis_residual_median=info['axis_residual_median'], kept=keep, canonicalized=bool(canonicalize), flipped=bool(flip),
                  outer_faces=int((labels == 1).sum()), inner_faces=int((labels == 2).sum()), break_faces=int((labels == 0).sum()))
    if checks: report['axis_check'] = axis_checks(V, F, labels)
    obs = {'schema_version': 'axis-development-0.1', 'coordinate_policy': 'registered_arbitrary_orientation', 'units': 'dimensionless', 'oracle_flags': [],
           'mesh': {'vertices': Vo.tolist(), 'faces': Fo.tolist()}}
    return obs, report, R


# ------------------------------------------------------------------------------------------------- CLI

def write_obj(path, V, F, report=None):
    import json
    with open(path, 'w') as fh:
        fh.write('# surviving outer wall prepared by project/intake/intake.py\n')
        if report is not None:      # travels with the file; the website reads this line on import
            keys = ('axis_check', 'wall_share', 'break_share', 'outer_faces', 'inner_faces', 'break_faces', 'kept', 'flipped')
            fh.write('# intake-report: ' + json.dumps({k: report[k] for k in keys if k in report}, separators=(',', ':')) + '\n')
        fh.writelines('v %.6f %.6f %.6f\n' % tuple(v) for v in V); fh.writelines('f %d %d %d\n' % (a + 1, b + 1, c + 1) for a, b, c in F)


def main(argv=None):
    import argparse, json, os
    ap = argparse.ArgumentParser(description='Prepare a scanned sherd (OBJ/PLY/STL) for the completion models.')
    ap.add_argument('input'); ap.add_argument('output', help='output OBJ (importable on the website, <= 8 MB)')
    ap.add_argument('--keep', choices=['outer', 'walls', 'all'], default='outer'); ap.add_argument('--max-vertices', type=int, default=90000)
    ap.add_argument('--no-canonicalize', action='store_true'); ap.add_argument('--flip', action='store_true', help='turn the sherd upside down after canonicalizing')
    a = ap.parse_args(argv)
    V, F = load_mesh(a.input); obs, rep, R = intake(V, F, keep=a.keep, max_vertices=a.max_vertices, canonicalize=not a.no_canonicalize, flip=a.flip)
    write_obj(a.output, np.asarray(obs['mesh']['vertices']), np.asarray(obs['mesh']['faces']), rep)
    rep.update(input=os.path.basename(a.input), output=os.path.basename(a.output), output_bytes=os.path.getsize(a.output), rotation_applied=R.tolist())
    json.dump(rep, open(os.path.splitext(a.output)[0] + '.report.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != 'rotation_applied'}, indent=1))
    if rep['output_bytes'] > 8_000_000: print('WARNING: output exceeds the website 8 MB import limit; lower --max-vertices')


if __name__ == '__main__':
    main()
