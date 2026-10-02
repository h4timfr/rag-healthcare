"""Report how full each column of each page is (bottom of last content vs. the bottom margin).

Usage: python column_fill.py file.pdf
"""
import sys

import pymupdf

pdf = pymupdf.open(sys.argv[1])
BOTTOM = 792 - 1.019 * 72  # bottom margin edge in points
for i, page in enumerate(pdf, start=1):
    items = []
    for b in page.get_text("blocks"):
        items.append((b[0], b[1], b[2], b[3]))
    for img in page.get_image_info():
        items.append(tuple(img["bbox"]))
    left = [it for it in items if (it[0] + it[2]) / 2 < 306]
    right = [it for it in items if (it[0] + it[2]) / 2 >= 306]
    def bottom(group):
        return max((g[3] for g in group), default=0)
    lb, rb = bottom(left), bottom(right)
    print(f"page {i}: left bottom {lb:6.1f} (free {(BOTTOM - lb) / 72:4.2f} in)  right bottom {rb:6.1f} (free {(BOTTOM - rb) / 72:4.2f} in)")
