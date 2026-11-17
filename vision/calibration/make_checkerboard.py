#!/usr/bin/env python3
"""Make a printable A4 checkerboard for camera calibration.

Default: 10 x 7 squares (9 x 6 inner corners), each square 25 mm.

  python3 make_checkerboard.py              # writes checkerboard_9x6_25mm.pdf
  python3 make_checkerboard.py --square-mm 20 --cols 9 --rows 6

Print it at 100% scale ("Actual size", NOT "Fit to page"), then measure one
square with a ruler. If it is not exactly 25 mm, use the size you measured
when you run calibrate_camera.py. Glue the page flat onto stiff cardboard.
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

DPI = 300
A4_MM = (210, 297)


def mm_to_px(mm):
    return int(round(mm / 25.4 * DPI))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cols", type=int, default=9, help="inner corners across")
    p.add_argument("--rows", type=int, default=6, help="inner corners down")
    p.add_argument("--square-mm", type=float, default=25.0)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    sq_x, sq_y = args.cols + 1, args.rows + 1
    # put the long side of the board along the long side of the page
    page_w, page_h = A4_MM if sq_x <= sq_y else A4_MM[::-1]
    board_w, board_h = sq_x * args.square_mm, sq_y * args.square_mm
    if board_w > page_w - 20 or board_h > page_h - 20:
        raise SystemExit("Board does not fit on A4 with a 10 mm border. Use smaller squares.")

    img = Image.new("L", (mm_to_px(page_w), mm_to_px(page_h)), 255)
    d = ImageDraw.Draw(img)
    x0 = (page_w - board_w) / 2
    y0 = (page_h - board_h) / 2
    for r in range(sq_y):
        for c in range(sq_x):
            if (r + c) % 2 == 0:
                left, top = x0 + c * args.square_mm, y0 + r * args.square_mm
                d.rectangle([mm_to_px(left), mm_to_px(top),
                             mm_to_px(left + args.square_mm) - 1, mm_to_px(top + args.square_mm) - 1], fill=0)
    label = (f"{args.cols} x {args.rows} inner corners, squares {args.square_mm:g} mm. "
             f"Print at 100% (actual size) and check one square with a ruler.")
    d.text((mm_to_px(10), mm_to_px(page_h - 8)), label, fill=0)

    out = Path(args.out or Path(__file__).with_name(
        f"checkerboard_{args.cols}x{args.rows}_{args.square_mm:g}mm.pdf"))
    img.save(out, "PDF", resolution=DPI)
    print("wrote", out)


if __name__ == "__main__":
    main()
