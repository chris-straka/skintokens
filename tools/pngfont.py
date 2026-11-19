#!/usr/bin/env python3
"""Pure-Python 8-bit PNG reader/writer and 5x7 bitmap font (stdlib only)."""

import os
import struct
import zlib

FONT_5X7 = {
    " ": [0, 0, 0, 0, 0, 0, 0],
    "-": [0, 0, 0, 31, 0, 0, 0],
    "?": [14, 17, 1, 2, 4, 0, 4],
    "/": [1, 2, 2, 4, 8, 8, 16],
    "(": [2, 4, 8, 8, 8, 4, 2],
    ")": [8, 4, 2, 2, 2, 4, 8],
    "_": [0, 0, 0, 0, 0, 0, 31],
    ":": [0, 4, 4, 0, 4, 4, 0],
    ".": [0, 0, 0, 0, 0, 4, 4],
    "'": [4, 4, 8, 0, 0, 0, 0],
    "0": [14, 17, 19, 21, 25, 17, 14],
    "1": [4, 12, 4, 4, 4, 4, 14],
    "2": [14, 17, 1, 2, 4, 8, 31],
    "3": [31, 2, 4, 2, 1, 17, 14],
    "4": [2, 6, 10, 18, 31, 2, 2],
    "5": [31, 16, 30, 1, 1, 17, 14],
    "6": [6, 8, 16, 30, 17, 17, 14],
    "7": [31, 1, 2, 4, 8, 8, 8],
    "8": [14, 17, 17, 14, 17, 17, 14],
    "9": [14, 17, 17, 15, 1, 2, 12],
    "A": [14, 17, 17, 31, 17, 17, 17],
    "B": [30, 17, 17, 30, 17, 17, 30],
    "C": [14, 17, 16, 16, 16, 17, 14],
    "D": [28, 18, 17, 17, 17, 18, 28],
    "E": [31, 16, 16, 30, 16, 16, 31],
    "F": [31, 16, 16, 30, 16, 16, 16],
    "G": [14, 17, 16, 23, 17, 17, 15],
    "H": [17, 17, 17, 31, 17, 17, 17],
    "I": [14, 4, 4, 4, 4, 4, 14],
    "J": [7, 2, 2, 2, 2, 18, 12],
    "K": [17, 18, 20, 24, 20, 18, 17],
    "L": [16, 16, 16, 16, 16, 16, 31],
    "M": [17, 27, 21, 21, 17, 17, 17],
    "N": [17, 25, 21, 19, 17, 17, 17],
    "O": [14, 17, 17, 17, 17, 17, 14],
    "P": [30, 17, 17, 30, 16, 16, 16],
    "Q": [14, 17, 17, 17, 21, 18, 13],
    "R": [30, 17, 17, 30, 20, 18, 17],
    "S": [15, 16, 16, 14, 1, 1, 30],
    "T": [31, 4, 4, 4, 4, 4, 4],
    "U": [17, 17, 17, 17, 17, 17, 14],
    "V": [17, 17, 17, 17, 17, 10, 4],
    "W": [17, 17, 17, 21, 21, 27, 17],
    "X": [17, 17, 10, 4, 10, 17, 17],
    "Y": [17, 17, 10, 4, 4, 4, 4],
    "Z": [31, 1, 2, 4, 8, 16, 31],
}


def read_png_rgb(path):
    """Reads an 8-bit RGB or RGBA non-interlaced PNG into `(w, h, bytearray_rgb)`."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path}: not a PNG")
    pos = 8
    w = h = ctype = 0
    idat = bytearray()
    while pos < len(data):
        ln, typ = struct.unpack(">I4s", data[pos:pos + 8])
        chunk = data[pos + 8:pos + 8 + ln]
        pos += 12 + ln
        if typ == b"IHDR":
            w, h, depth, ctype, comp, filt, interlace = struct.unpack(">IIBBBBB", chunk)
            if depth != 8 or ctype not in (2, 6) or interlace != 0:
                raise ValueError(f"{path}: unsupported PNG format depth={depth} ctype={ctype}")
        elif typ == b"IDAT":
            idat.extend(chunk)
        elif typ == b"IEND":
            break
    raw = zlib.decompress(bytes(idat))
    bpp = 3 if ctype == 2 else 4
    stride = w * bpp
    out = bytearray(w * h * 3)
    prev = bytearray(stride)
    cur = bytearray(stride)
    src = 0
    for y in range(h):
        ft = raw[src]
        src += 1
        row = raw[src:src + stride]
        src += stride
        if ft == 0:
            cur[:] = row
        elif ft == 1:
            for i in range(stride):
                cur[i] = (row[i] + (cur[i - bpp] if i >= bpp else 0)) & 0xFF
        elif ft == 2:
            for i in range(stride):
                cur[i] = (row[i] + prev[i]) & 0xFF
        elif ft == 3:
            for i in range(stride):
                a = cur[i - bpp] if i >= bpp else 0
                b = prev[i]
                cur[i] = (row[i] + ((a + b) >> 1)) & 0xFF
        elif ft == 4:
            for i in range(stride):
                a = cur[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                cur[i] = (row[i] + pr) & 0xFF
        else:
            raise ValueError(f"bad PNG filter {ft}")
        if bpp == 3:
            out[y * w * 3:(y + 1) * w * 3] = cur
        else:
            dst = y * w * 3
            for x in range(w):
                out[dst + x * 3:dst + x * 3 + 3] = cur[x * 4:x * 4 + 3]
        prev[:] = cur
    return w, h, out


def write_png_rgb(path, w, h, rgb):
    def chunk(typ, body):
        crc = zlib.crc32(typ + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + typ + body + struct.pack(">I", crc)

    raw = bytearray()
    stride = w * 3
    for y in range(h):
        raw.append(0)
        raw.extend(rgb[y * stride:(y + 1) * stride])
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(png)


def draw_text(buf, w, h, x0, y0, text, scale, color):
    r, g, b = color
    cx = x0
    for ch in text.upper():
        rows = FONT_5X7.get(ch, FONT_5X7["?"])
        for ry, bits in enumerate(rows):
            for rx in range(5):
                if (bits >> (4 - rx)) & 1:
                    for sy in range(scale):
                        py = y0 + ry * scale + sy
                        if 0 <= py < h:
                            for sx in range(scale):
                                px = cx + rx * scale + sx
                                if 0 <= px < w:
                                    o = (py * w + px) * 3
                                    buf[o] = r
                                    buf[o + 1] = g
                                    buf[o + 2] = b
        cx += 6 * scale


def text_width(text, scale):
    return max(0, len(text) * 6 * scale - scale)
