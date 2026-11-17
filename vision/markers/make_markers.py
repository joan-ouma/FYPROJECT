#!/usr/bin/env python3
"""Make printable ArUco markers (A4 PDF) for the shelf slots.

  python3 markers/make_markers.py                 # markers 0-4, 40 mm, DICT_4X4_50
  python3 markers/make_markers.py --ids 0 1 2 3 4 10 --size-mm 50

Print at 100% ("Actual size"). Measure the BLACK square of one marker with a ruler
and put that number in config/vision_config.yaml -> aruco.marker_size_mm.
Leave the white border around each marker when you cut it out: the detector needs it.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shopvision.aruco import get_dictionary  # noqa: E402

DPI = 300
A4_MM = (210, 297)


def px(mm):
    return int(round(mm / 25.4 * DPI))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--size-mm", type=float, default=40.0, help="size of the black square")
    p.add_argument("--dictionary", default="DICT_4X4_50")
    p.add_argument("--out", default=str(Path(__file__).with_name("aruco_markers.pdf")))
    args = p.parse_args(argv)

    d = get_dictionary(args.dictionary)
    border = max(8.0, args.size_mm * 0.25)          # white margin around each marker
    cell = args.size_mm + 2 * border + 8            # + room for the label
    cols = int((A4_MM[0] - 20) // cell)
    rows = int((A4_MM[1] - 20) // cell)
    if cols < 1 or rows < 1:
        raise SystemExit("Marker too big for A4")
    per_page = cols * rows
    pages = []
    for start in range(0, len(args.ids), per_page):
        page = Image.new("L", (px(A4_MM[0]), px(A4_MM[1])), 255)
        draw = ImageDraw.Draw(page)
        for k, mid in enumerate(args.ids[start:start + per_page]):
            r, c = divmod(k, cols)
            x0 = 10 + c * cell
            y0 = 10 + r * cell
            side = px(args.size_mm)
            marker = cv2.aruco.generateImageMarker(d, mid, side)
            page.paste(Image.fromarray(marker), (px(x0 + border), px(y0 + border)))
            # thin grey cut line around the white border
            draw.rectangle([px(x0), px(y0), px(x0 + args.size_mm + 2 * border),
                            px(y0 + args.size_mm + 2 * border)], outline=170, width=2)
            draw.text((px(x0 + border), px(y0 + args.size_mm + 2 * border + 1.5)),
                      f"ID {mid}  ({args.dictionary}, {args.size_mm:g} mm)", fill=0)
        pages.append(page)
    pages[0].save(args.out, "PDF", resolution=DPI, save_all=True, append_images=pages[1:])
    print(f"wrote {args.out}: {len(args.ids)} markers on {len(pages)} page(s)")


if __name__ == "__main__":
    main()
