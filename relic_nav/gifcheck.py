"""Spec-compliant GIF verifier used by the test suite.

Parses every block (including graphic control extensions) and LZW-decodes
every image, confirming each frame yields exactly width*height indices.
Pure standard library; read-only."""

import struct


class BitReader:
    def __init__(self, blob):
        self.blob = blob
        self.bi = 0
        self.bits = 0
        self.nb = 0

    def read(self, n):
        while self.nb < n:
            if self.bi >= len(self.blob):
                raise EOFError("ran out of LZW bytes")
            self.bits |= self.blob[self.bi] << self.nb
            self.nb += 8
            self.bi += 1
        c = self.bits & ((1 << n) - 1)
        self.bits >>= n
        self.nb -= n
        return c


def lzw_decode(blob, min_code_size, want):
    """Decode one GIF LZW data stream; raises unless exactly want pixels."""
    clear = 1 << min_code_size
    end = clear + 1
    size = min_code_size + 1
    table = {i: bytes([i]) for i in range(clear)}
    nxt = end + 1
    out = bytearray()
    br = BitReader(blob)
    prev = None
    while True:
        c = br.read(size)
        if c == clear:
            size = min_code_size + 1
            table = {i: bytes([i]) for i in range(clear)}
            nxt = end + 1
            prev = None
            continue
        if c == end:
            break
        if prev is None:
            out.extend(table[c])
            prev = c
            continue
        if c in table:
            entry = table[c]
            table[nxt] = table[prev] + entry[:1]
        else:
            if c != nxt:
                raise ValueError(f"invalid code {c}, expected <= {nxt}")
            entry = table[prev] + table[prev][:1]
            table[nxt] = entry
        out.extend(entry)
        nxt += 1
        if nxt == (1 << size) and size < 12:
            size += 1
        prev = c
        if len(out) > want:
            raise ValueError(f"decoded {len(out)} > {want} pixels")
    if len(out) != want:
        raise ValueError(f"decoded {len(out)} != {want} pixels")
    return out


def _sub_blocks(d, p):
    payload = bytearray()
    while True:
        size = d[p]
        p += 1
        if size == 0:
            return bytes(payload), p
        payload.extend(d[p:p + size])
        p += size


def verify(path):
    """Return (width, height, frame_count); raise on any malformed block."""
    with open(path, "rb") as fh:
        d = fh.read()
    if d[:6] not in (b"GIF87a", b"GIF89a"):
        raise ValueError("bad GIF signature")
    w, h = struct.unpack("<HH", d[6:10])
    packed = d[10]
    if not packed & 0x80:
        raise ValueError("global color table required")
    gct_n = 2 ** ((packed & 7) + 1)
    p = 13 + 3 * gct_n
    frames = 0
    while True:
        tag = d[p]
        if tag == 0x3B:
            p += 1
            break
        if tag == 0x21:
            label = d[p + 1]
            p += 2
            if label == 0xF9:  # graphic control: one fixed 4-byte block
                if d[p] != 4:
                    raise ValueError("malformed graphic control block")
                p += 4
                if d[p] != 0:
                    raise ValueError("missing GCE terminator")
                p += 1
            else:
                _payload, p = _sub_blocks(d, p)
        elif tag == 0x2C:
            _il, _it, iw, ih = struct.unpack("<HHHH", d[p + 1:p + 9])
            ip = d[p + 9]
            p += 10
            if ip & 0x80:  # local color table
                p += 3 * 2 ** ((ip & 7) + 1)
            mcs = d[p]
            p += 1
            blob, p = _sub_blocks(d, p)
            lzw_decode(blob, mcs, iw * ih)
            frames += 1
        else:
            raise ValueError(f"unknown block tag 0x{tag:02x} at {p}")
    if p != len(d):
        raise ValueError(f"trailing {len(d) - p} bytes")
    return w, h, frames
