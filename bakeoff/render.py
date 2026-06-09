#!/usr/bin/env python3
"""Render sample.pdf -> page PNGs (CPU, reproducible).

The source is a CCITT-G4 (Group 4) black-and-white scan: each page is one
embedded fax stream with no text layer. We extract each stream verbatim, wrap
it in a minimal TIFF header (no decoding by hand), and let Pillow/libtiff
decode to PNG. Deterministic: same PDF in -> identical PNGs out.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import struct
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_ccitt_images(pdf_bytes: bytes):
    """Yield (width, height, columns, raw_g4_bytes) per image XObject."""
    images = []
    for m in re.finditer(rb"/Subtype\s*/Image", pdf_bytes):
        dict_start = pdf_bytes.rfind(b"<<", 0, m.start())
        stream_kw = pdf_bytes.find(b"stream", m.start())
        blob = pdf_bytes[dict_start:stream_kw]

        def grab(key: bytes):
            mm = re.search(key + rb"\s+(\d+)", blob)
            return int(mm.group(1)) if mm else None

        w, h = grab(b"/Width"), grab(b"/Height")
        if w is None or h is None:
            continue
        cm = re.search(rb"/Columns\s+(\d+)", blob)
        cols = int(cm.group(1)) if cm else w
        lm = re.search(rb"/Length\s+(\d+)", blob)
        length = int(lm.group(1)) if lm else None

        p = stream_kw + len(b"stream")
        if pdf_bytes[p : p + 2] == b"\r\n":
            p += 2
        elif pdf_bytes[p : p + 1] in (b"\n", b"\r"):
            p += 1
        if length is not None:
            raw = pdf_bytes[p : p + length]
        else:
            end = pdf_bytes.find(b"endstream", p)
            raw = pdf_bytes[p:end].rstrip(b"\r\n")
        images.append((w, h, cols, raw))
    return images


def wrap_g4_tiff(width: int, height: int, raw: bytes) -> bytes:
    """Minimal little-endian TIFF, Compression=4 (CCITT G4), Photometric=0."""
    entries = [
        (256, 4, 1, width), (257, 4, 1, height), (258, 3, 1, 1), (259, 3, 1, 4),
        (262, 3, 1, 0), (273, 4, 1, 8), (278, 4, 1, height), (279, 4, 1, len(raw)),
    ]
    header = b"II" + struct.pack("<HI", 42, 8 + len(raw))
    ifd = struct.pack("<H", len(entries))
    for tag, typ, cnt, val in entries:
        if typ == 3:  # SHORT, left-justified in the 4-byte value field
            ifd += struct.pack("<HHI", tag, typ, cnt) + struct.pack("<HH", val, 0)
        else:  # LONG
            ifd += struct.pack("<HHII", tag, typ, cnt, val)
    ifd += struct.pack("<I", 0)
    return header + raw + ifd


def main() -> int:
    ap = argparse.ArgumentParser(description="Render sample.pdf to page PNGs")
    ap.add_argument("--config", default=str(ROOT / "bakeoff" / "config.json"))
    ap.add_argument("--out", default=None, help="override output dir (default: paths.pages)")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    pdf_path = ROOT / cfg["paths"]["pdf"]
    out_dir = Path(args.out) if args.out else ROOT / cfg["paths"]["pages"]

    if not pdf_path.exists():
        sys.exit(f"render: input PDF not found: {pdf_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    images = extract_ccitt_images(pdf_path.read_bytes())
    if not images:
        sys.exit("render: no image XObjects found — is this the expected scan?")

    for i, (w, h, cols, raw) in enumerate(images, 1):
        tif = wrap_g4_tiff(cols, h, raw)
        im = Image.open(io.BytesIO(tif))
        im.load()
        out = out_dir / f"page{i}.png"
        im.convert("L").save(out)

    print(f"ok: wrote {len(images)} pages to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
