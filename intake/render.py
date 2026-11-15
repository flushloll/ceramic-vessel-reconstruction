"""Tiny dependency-free renderer for checking segmentation by eye: splats face centroids with Lambert shading
and per-label colours into a z-buffer, writes PNG with zlib. Not used by the pipeline itself."""
import struct, zlib
import numpy as np

LABEL_RGB = {1: (46, 120, 214), 2: (27, 175, 122), 0: (227, 73, 72)}   # outer blue, inner aqua, break red


def _png(path, img):
    h, w, _ = img.shape; raw = b''.join(b'\x00' + img[y].tobytes() for y in range(h))
    def chunk(t, d): return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    open(path, 'wb').write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def render_points(V, F, points, path, views=((0, 15), (90, 15), (180, 15), (0, 80)), size=360, point_rgb=(40, 40, 40), surface_rgb=(196, 186, 170)):
    """Sherd surface (shaded) plus predicted points (dark dots) - for looking at predictions on real sherds."""
    allp = np.concatenate([V[F].reshape(-1, 3), points]); c0 = allp.mean(0); scale = np.abs(allp - c0).max() * 1.05
    t = V[F]; cent = t.mean(1) - c0; cr = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]); n = cr / np.maximum(np.linalg.norm(cr, axis=1)[:, None], 1e-12); pts = points - c0; tiles = []
    for yaw, pitch in views:
        a, b = np.radians(yaw), np.radians(pitch)
        Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]); Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]]); R = Rx @ Ry
        img = np.full((size, size, 3), 250, np.uint8)
        def px(p): return ((p[:, 0] / scale + 1) / 2 * (size - 1)).astype(int).clip(0, size - 1), ((1 - (p[:, 1] / scale + 1) / 2) * (size - 1)).astype(int).clip(0, size - 1)
        p = cent @ R.T; o = np.argsort(p[:, 2]); x, y = px(p); sh = (0.35 + 0.65 * np.abs((n @ R.T)[:, 2]))[:, None] * np.array(surface_rgb)
        for dx in (0, 1):
            for dy in (0, 1): img[(y + dy).clip(0, size - 1)[o], (x + dx).clip(0, size - 1)[o]] = sh[o].astype(np.uint8)
        q = pts @ R.T; x, y = px(q)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1): img[(y + dy).clip(0, size - 1), (x + dx).clip(0, size - 1)] = point_rgb
        tiles.append(img)
    _png(path, np.concatenate(tiles, 1))


def render(V, F, labels, path, views=((0, 0), (90, 0), (0, 90), (180, 30)), size=360):
    t = V[F]; cent = t.mean(1); cr = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]); n = cr / np.maximum(np.linalg.norm(cr, axis=1)[:, None], 1e-12)
    cent = cent - cent.mean(0); scale = np.abs(cent).max() * 1.1; tiles = []
    for yaw, pitch in views:
        a, b = np.radians(yaw), np.radians(pitch)
        Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]); Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
        R = Rx @ Ry; p = cent @ R.T; nn = n @ R.T
        x = ((p[:, 0] / scale + 1) / 2 * (size - 1)).astype(int); y = ((1 - (p[:, 1] / scale + 1) / 2) * (size - 1)).astype(int)
        img = np.full((size, size, 3), 250, np.uint8); zb = np.full((size, size), -np.inf)
        shade = 0.35 + 0.65 * np.abs(nn[:, 2]); col = np.array([LABEL_RGB[int(l)] for l in labels]) * shade[:, None]
        order = np.argsort(p[:, 2])
        for dx in (0, 1):
            for dy in (0, 1):
                xx, yy = np.clip(x + dx, 0, size - 1), np.clip(y + dy, 0, size - 1)
                img[yy[order], xx[order]] = col[order].astype(np.uint8)      # later (nearer) faces overwrite
        tiles.append(img)
    _png(path, np.concatenate(tiles, 1))
