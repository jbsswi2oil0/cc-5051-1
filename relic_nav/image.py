"""RGB framebuffer with raster drawing primitives and pure-stdlib PNG/GIF
writers. GIF uses a fixed 256-colour palette with nearest-colour mapping."""

import struct
import zlib

from .font5x7 import draw_text, text_width


class Canvas:
    def __init__(self, w, h, bg=(18, 18, 24)):
        self.w = w
        self.h = h
        self.pixels = [tuple(bg)] * (w * h)

    def clear(self, bg):
        self.pixels = [tuple(bg)] * (self.w * self.h)

    def put(self, x, y, c):
        if 0 <= x < self.w and 0 <= y < self.h:
            self.pixels[y * self.w + x] = tuple(c)

    def fill_rect(self, x0, y0, x1, y1, c):
        x0 = max(0, int(x0)); y0 = max(0, int(y0))
        x1 = min(self.w, int(x1)); y1 = min(self.h, int(y1))
        c = tuple(c)
        for y in range(y0, y1):
            base = y * self.w
            for x in range(x0, x1):
                self.pixels[base + x] = c

    def line(self, p0, p1, c, width=1):
        x0, y0 = p0
        x1, y1 = p1
        dx = x1 - x0
        dy = y1 - y0
        steps = max(1, int(max(abs(dx), abs(dy))))
        half = width / 2.0
        r = int((width + 1) // 2)
        for i in range(steps + 1):
            t = i / steps
            x = x0 + dx * t
            yy = y0 + dy * t
            if width <= 1:
                self.put(int(round(x)), int(round(yy)), c)
            else:
                ix, iy = int(round(x)), int(round(yy))
                for ox in range(-r, r + 1):
                    for oy in range(-r, r + 1):
                        if abs(ox) <= half and abs(oy) <= half:
                            self.put(ix + ox, iy + oy, c)

    def polygon(self, pts, fill=None, outline=None, width=1):
        if fill is not None:
            ys = [p[1] for p in pts]
            for y in range(max(0, int(min(ys))), min(self.h, int(max(ys)) + 1)):
                xs = []
                n = len(pts)
                for i in range(n):
                    ax, ay = pts[i]
                    bx, by = pts[(i + 1) % n]
                    if (ay > y) != (by > y):
                        xs.append(ax + (y - ay) * (bx - ax) / (by - ay))
                xs.sort()
                for k in range(0, len(xs) - 1, 2):
                    self.fill_rect(int(xs[k]) + 1, y, int(xs[k + 1]), y + 1, fill)
        if outline is not None:
            for a, b in zip(pts, pts[1:] + pts[:1]):
                self.line(a, b, outline, width)

    def circle(self, cx, cy, r, fill=None, outline=None, width=1):
        r = int(round(r))
        for y in range(-r, r + 1):
            for x in range(-r, r + 1):
                d2 = x * x + y * y
                if fill is not None and d2 <= r * r:
                    self.put(cx + x, cy + y, fill)
        if outline is not None:
            rr = r
            for a in range(360):
                import math
                ang = math.radians(a)
                x = cx + rr * math.cos(ang)
                yy = cy + rr * math.sin(ang)
                self.put(int(round(x)), int(round(yy)), outline)
                if width > 1:
                    for ox in range(-width // 2, width // 2 + 1):
                        for oy in range(-width // 2, width // 2 + 1):
                            self.put(int(round(x)) + ox, int(round(yy)) + oy, outline)

    def text(self, s, x, y, c, scale=1):
        draw_text(self.pixels, self.w, self.h, s, x, y, tuple(c), scale=scale)

    def text_width(self, s, scale=1):
        return text_width(s, scale=scale)

    def palette_indices(self):
        cache = {}
        out = bytearray(len(self.pixels))
        for k, c in enumerate(self.pixels):
            idx = cache.get(c)
            if idx is None:
                idx = nearest_index(c)
                cache[c] = idx
            out[k] = idx
        return out

    def save_png(self, path):
        raw = bytearray()
        for y in range(self.h):
            raw.append(0)  # filter type 0
            base = y * self.w
            for x in range(self.w):
                raw.extend(self.pixels[base + x])
        png = bytearray(b"\x89PNG\r\n\x1a\n")

        def chunk(tag, data):
            png.extend(struct.pack(">I", len(data)))
            png.extend(tag)
            png.extend(data)
            png.extend(struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))

        chunk(b"IHDR", struct.pack(">IIBBBBB", self.w, self.h, 8, 2, 0, 0, 0))
        chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        chunk(b"IEND", b"")
        with open(path, "wb") as f:
            f.write(png)


# ------------------------------------------------------------------- GIF output
# 6x7x6 RGB cube + greys, 252..255 reserved
def build_palette():
    pal = []
    for r in range(6):
        for g in range(7):
            for b in range(6):
                pal.append((r * 51, g * 42, b * 51))
    # 252 entries used above (6*7*6); fill remainder with greys
    greys = [0, 32, 64, 96, 128, 160, 192, 224, 255]
    i = 252
    for gv in greys:
        if i >= 256:
            break
        pal.append((gv, gv, gv))
        i += 1
    while len(pal) < 256:
        pal.append((255, 0, 255))
    return pal


PALETTE = build_palette()
_NCACHE = {}


def nearest_index(c):
    r, g, b = c
    key = (r >> 3, g >> 3, b >> 3)
    hit = _NCACHE.get(key)
    if hit is not None:
        return hit
    best, bd = 0, 1 << 30
    for i, (pr, pg, pb) in enumerate(PALETTE):
        d = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2
        if d < bd:
            bd, best = d, i
    _NCACHE[key] = best
    return best


def _lzw_compress(indices, min_code_size):
    clear_code = 1 << min_code_size
    end_code = clear_code + 1
    code_size = min_code_size + 1
    dict_next = end_code + 1
    out_bits = 0
    n_bits = 0
    packed = bytearray()

    def emit(code):
        nonlocal out_bits, n_bits
        out_bits |= code << n_bits
        n_bits += code_size
        while n_bits >= 8:
            packed.append(out_bits & 0xff)
            out_bits >>= 8
            n_bits -= 8

    # dictionary: (prefix_code, symbol) -> new code. Literal codes 0..clear-1
    # are implicit; the table never stores byte sequences, so every step is
    # O(1) regardless of phrase length.
    table = {}
    emit(clear_code)
    prev = indices[0] if indices else 0
    for val in indices[1:]:
        key = (prev, val)
        nxt = table.get(key)
        if nxt is not None:
            prev = nxt
            continue
        emit(prev)
        if dict_next < 4096:
            table[key] = dict_next
            dict_next += 1
            # GIF LZW: the width grows once the slot past 2^width is assigned
            if dict_next == (1 << code_size) + 1 and code_size < 12:
                code_size += 1
        else:
            emit(clear_code)
            table = {}
            dict_next = end_code + 1
            code_size = min_code_size + 1
        prev = val
    emit(prev)
    emit(end_code)
    if n_bits > 0:
        packed.append(out_bits & 0xff)
    return bytes(packed)


def save_gif(canvases, path, delay_cs=5):
    """Write an animated GIF89a (one global palette, no transparency)."""
    w, h = canvases[0].w, canvases[0].h
    min_code = 8
    data = bytearray()
    data.extend(b"GIF89a")
    data.extend(struct.pack("<HH", w, h))
    data.append(0xf0 | (min_code - 1))  # global color table, size 256
    data.append(0)
    data.append(0)
    for c in PALETTE:
        data.extend(c)
    # netscape loop
    data.extend(b"!\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00")
    for canvas in canvases:
        indices = canvas.palette_indices()
        data.extend(b"\x21\xf9\x04\x00")
        data.extend(struct.pack("<H", delay_cs))
        data.extend(b"\x00")
        data.extend(b"\x2c")
        data.extend(struct.pack("<HHHH", 0, 0, w, h))
        data.append(0)
        data.append(min_code)
        compressed = _lzw_compress(indices, min_code)
        # sub-blocks of <=255 bytes
        for i in range(0, len(compressed), 255):
            block = compressed[i:i + 255]
            data.append(len(block))
            data.extend(block)
        data.append(0)
    data.append(0x3b)
    with open(path, "wb") as f:
        f.write(data)
