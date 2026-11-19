"""PNG/GIF writers: file structure and LZW round-trip integrity."""

import os
import struct
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relic_nav.image import Canvas, save_gif, _lzw_compress, PALETTE
from relic_nav.gifcheck import verify, lzw_decode


class TestPNG(unittest.TestCase):
    def test_png_round_trip_pixels(self):
        cv = Canvas(7, 5, bg=(10, 20, 30))
        cv.fill_rect(1, 1, 4, 4, (200, 100, 50))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.png")
            cv.save_png(path)
            with open(path, "rb") as fh:
                data = fh.read()
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        # parse chunks and decode IDAT
        pos = 8
        idat = b""
        width = height = None
        while pos < len(data):
            length = struct.unpack(">I", data[pos:pos + 4])[0]
            tag = data[pos + 4:pos + 8]
            body = data[pos + 8:pos + 8 + length]
            if tag == b"IHDR":
                width, height, depth, color = struct.unpack(">IIBB", body[:10])
            elif tag == b"IDAT":
                idat += body
            pos += 12 + length
        self.assertEqual((width, height), (7, 5))
        raw = zlib.decompress(idat)
        stride = 1 + width * 3
        self.assertEqual(len(raw), stride * height)
        # filter byte of every scanline must be 0 (the encoder only emits None)
        for y in range(height):
            self.assertEqual(raw[y * stride], 0)
        # spot-check the filled rectangle pixel
        px = raw[1 * stride + 1 + 2 * 3: 1 * stride + 1 + 3 * 3]
        self.assertEqual(tuple(px), (200, 100, 50))


class TestLZW(unittest.TestCase):
    def test_round_trip_all_min_code_sizes(self):
        import random
        rng = random.Random(99)
        for mcs in (2, 4, 8):
            n = 1 << mcs
            src = bytes(rng.randrange(n) for _ in range(3000))
            blob = _lzw_compress(src, mcs)
            out = lzw_decode(blob, mcs, len(src))
            self.assertEqual(bytes(out), src)

    def test_dictionary_reset_path(self):
        # enough symbols to exhaust the 4096-entry table
        import random
        rng = random.Random(7)
        src = bytes(rng.randrange(256) for _ in range(120000))
        blob = _lzw_compress(src, 8)
        out = lzw_decode(blob, 8, len(src))
        self.assertEqual(bytes(out), src)

    def test_constant_stream(self):
        src = b"\x07" * 20000
        out = lzw_decode(_lzw_compress(src, 8), 8, len(src))
        self.assertEqual(bytes(out), src)


class TestGIFContainer(unittest.TestCase):
    def test_single_and_multi_frame_gifs(self):
        with tempfile.TemporaryDirectory() as tmp:
            one = [Canvas(13, 9)]
            p1 = os.path.join(tmp, "one.gif")
            save_gif(one, p1)
            w, h, frames = verify(p1)
            self.assertEqual((w, h, frames), (13, 9, 1))

            frames_src = []
            for k in range(5):
                c = Canvas(13, 9)
                c.fill_rect(0, 0, k + 1, 9, PALETTE[(k * 17) % 256])
                frames_src.append(c)
            p2 = os.path.join(tmp, "multi.gif")
            save_gif(frames_src, p2)
            w, h, frames = verify(p2)
            self.assertEqual((w, h, frames), (13, 9, 5))

    def test_palette_has_256_entries(self):
        self.assertEqual(len(PALETTE), 256)
        for r, g, b in PALETTE:
            self.assertTrue(0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255)


if __name__ == "__main__":
    unittest.main()
