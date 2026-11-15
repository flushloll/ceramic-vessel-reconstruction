"""Unit tests for the intake pipeline on a synthetic sherd with a known answer. Run: python -I -B project/intake/test_intake.py"""
import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intake as I


def thick_sherd(r=1.0, t=0.08, h=(0.0, 1.2), ang=(0.0, 1.4), na=40, nh=30):
    """Curved piece of a cylinder wall (axis = +Y through origin): outer + inner wall + 4 break faces, outward winding."""
    A = np.linspace(*ang, na); Hh = np.linspace(*h, nh); V = []; F = []
    def grid(rad):
        base = len(V)
        for y in Hh:
            for a in A: V.append([rad * np.cos(a), y, rad * np.sin(a)])
        return base
    o = grid(r); i = grid(r - t); idx = lambda b, j, k: b + j * na + k
    for j in range(nh - 1):
        for k in range(na - 1):
            a, b, c, d = idx(o, j, k), idx(o, j, k + 1), idx(o, j + 1, k), idx(o, j + 1, k + 1)
            F += [[a, c, b], [b, c, d]]                                   # outer: normal points away from axis
            a, b, c, d = idx(i, j, k), idx(i, j, k + 1), idx(i, j + 1, k), idx(i, j + 1, k + 1)
            F += [[a, b, c], [b, d, c]]                                   # inner: normal points toward axis
    for j in range(nh - 1):                                               # two vertical break faces
        for k in (0, na - 1):
            a, b, c, d = idx(o, j, k), idx(o, j + 1, k), idx(i, j, k), idx(i, j + 1, k); F += [[a, b, c], [b, d, c]]
    for k in range(na - 1):                                               # top and bottom break faces
        for j in (0, nh - 1):
            a, b, c, d = idx(o, j, k), idx(o, j, k + 1), idx(i, j, k), idx(i, j, k + 1); F += [[a, b, c], [b, d, c]]
    return np.asarray(V, float), np.asarray(F, np.int64)


def random_rotation(seed):
    q = np.random.default_rng(seed).normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def test_segmentation_and_axis():
    V, F = thick_sherd(); Rr = random_rotation(3); Vr = V @ Rr.T + np.array([5.0, -2.0, 7.0])
    labels, info = I.segment(*I.weld(Vr, F))
    true_axis = Rr @ np.array([0, 1, 0.0])
    err = np.degrees(np.arccos(min(1, abs(info['axis_direction'] @ true_axis))))
    assert err < 1.0, f'axis error {err:.3f} deg'
    n_wall_faces = 2 * (40 - 1) * (30 - 1) * 2 // 2
    assert (labels == 1).sum() == n_wall_faces and (labels == 2).sum() == n_wall_faces, (labels == 1).sum()
    assert (labels == 0).sum() == len(F) - 2 * n_wall_faces
    print('segmentation_and_axis OK  axis error %.4f deg, break share %.3f' % (err, info['break_share']))


def barrel_sherd(t=0.06, na=40, nh=40):
    """Curved-profile (barrel) sherd: radius varies with height, so the line-complex (non-degenerate) path is used."""
    V, F = thick_sherd(t=t, na=na, nh=nh)
    y = V[:, 1]; rad = np.linalg.norm(V[:, [0, 2]], axis=1); scale = (1.0 + 0.35 * np.sin(np.pi * y / 1.2)) * rad / np.maximum(rad, 1e-12)
    V = V.copy(); V[:, 0] *= scale; V[:, 2] *= scale; return V, F


def test_barrel_axis_and_segmentation():
    V, F = barrel_sherd(); Rr = random_rotation(7); Vr = V @ Rr.T - np.array([3.0, 1.0, 2.0])
    labels, info = I.segment(*I.weld(Vr, F)); true_axis = Rr @ np.array([0, 1, 0.0])
    err = np.degrees(np.arccos(min(1, abs(info['axis_direction'] @ true_axis))))
    assert err < 3.0, f'barrel axis error {err:.3f} deg'   # faceting of a curved surface limits accuracy (~2 deg here)
    assert (labels == 1).sum() == 3042 and (labels == 2).sum() == 3042 and (labels == 0).sum() == 312, [(labels == k).sum() for k in (1, 2, 0)]
    print('barrel axis/segmentation OK  axis error %.4f deg; outer %d inner %d break %d' % (err, (labels == 1).sum(), (labels == 2).sum(), (labels == 0).sum()))


def test_intake_canonicalizes_and_keeps_outer():
    V, F = thick_sherd(); Rr = random_rotation(11); Vr = V @ Rr.T
    obs, rep, R = I.intake(Vr, F)
    Vo = np.asarray(obs['mesh']['vertices'])
    # the kept outer wall must now be a surface of revolution about a vertical axis: radius constant per height
    cent, nrm, area = I.face_geometry(Vo, np.asarray(obs['mesh']['faces']))
    d, c, res = I.line_complex_axis(cent, nrm, area)
    assert abs(abs(d[1]) - 1) < 1e-3, d
    assert rep['inner_faces'] > 0 and rep['break_faces'] > 0 and rep['faces_kept'] == rep['outer_faces']
    print('intake_canonicalizes OK  kept %d outer of %d faces' % (rep['faces_kept'], rep['faces_in']))


def test_loaders_roundtrip():
    V, F = thick_sherd(na=8, nh=5); d = tempfile.mkdtemp()
    with open(os.path.join(d, 'm.obj'), 'w') as fh:
        fh.writelines(f'v {x} {y} {z}\n' for x, y, z in V); fh.writelines(f'f {a+1}/1/1 {b+1}/1/1 {c+1}/1/1\n' for a, b, c in F)
    V2, F2 = I.load_mesh(os.path.join(d, 'm.obj')); assert np.allclose(V2[F2], V[F])
    with open(os.path.join(d, 'm.ply'), 'w') as fh:
        fh.write(f'ply\nformat ascii 1.0\nelement vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\nelement face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n')
        fh.writelines(f'{x} {y} {z}\n' for x, y, z in V); fh.writelines(f'3 {a} {b} {c}\n' for a, b, c in F)
    V3, F3 = I.load_mesh(os.path.join(d, 'm.ply')); assert np.allclose(V3[F3], V[F])
    import struct
    with open(os.path.join(d, 'm.stl'), 'wb') as fh:
        fh.write(b'\0' * 80 + struct.pack('<I', len(F)))
        for tri in V[F]: fh.write(struct.pack('<12fH', 0, 0, 0, *tri.reshape(-1), 0))
    V4, F4 = I.load_mesh(os.path.join(d, 'm.stl')); assert np.allclose(V4[F4], V[F], atol=1e-6)
    print('loaders OK (OBJ, ASCII PLY, binary STL)')


def test_decimate_keeps_shape():
    V, F = thick_sherd(na=200, nh=150); V2, F2 = I.decimate(*I.weld(V, F), max_vertices=5000)
    assert len(V2) <= 5000 and len(F2) > 1000
    r = np.linalg.norm(V2[:, [0, 2]], axis=1); assert r.min() > 0.9 - 1e-6 and r.max() < 1.0 + 1e-6
    print('decimate OK  %d -> %d vertices' % (len(V), len(V2)))


def test_axis_checks_verdicts():
    V, F = barrel_sherd(); Vw, Fw = I.weld(V @ random_rotation(5).T, F); lab, _ = I.segment(Vw, Fw)
    good = I.axis_checks(Vw, Fw, lab); assert good['verdict'] == 'well determined', good
    xs, zs = np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, 1, 40)); Vp = np.stack([xs.ravel(), 0.002 * np.sin(9 * xs.ravel()), zs.ravel()], 1)
    Fp = np.array([[i * 40 + j, (i + 1) * 40 + j, i * 40 + j + 1] for i in range(39) for j in range(39)] + [[i * 40 + j + 1, (i + 1) * 40 + j, (i + 1) * 40 + j + 1] for i in range(39) for j in range(39)])
    flat = I.axis_checks(Vp, Fp); assert flat['verdict'] == 'undetermined', flat
    print('axis_checks OK  barrel:', good, ' near-flat plate:', flat['verdict'])


if __name__ == '__main__':
    test_axis_checks_verdicts()
    test_loaders_roundtrip(); test_segmentation_and_axis(); test_barrel_axis_and_segmentation(); test_intake_canonicalizes_and_keeps_outer(); test_decimate_keeps_shape(); print('ALL PASS')
