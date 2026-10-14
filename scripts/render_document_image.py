"""Render a text document to a PNG so the Textract path can be exercised: render_document_image.py in.txt out.png"""
import sys
import textwrap

from PIL import Image, ImageDraw, ImageFont

src, dst = sys.argv[1], sys.argv[2]
lines = [w for para in open(src, encoding="utf-8").read().splitlines() for w in (textwrap.wrap(para, 70) or [""])]
try:
    font = ImageFont.truetype("arial.ttf", 22)
except OSError:
    font = ImageFont.load_default()
img = Image.new("RGB", (1100, 40 + 34 * len(lines)), "white")
draw = ImageDraw.Draw(img)
for i, line in enumerate(lines):
    draw.text((30, 20 + 34 * i), line, fill="black", font=font)
img.save(dst)
