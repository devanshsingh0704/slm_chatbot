"""Checks that ingest.extract() reads every kind of file, using OCR where a page has no text layer.
Usage: python test_ocr.py   (makes its own test files in a temporary folder; needs rapidocr)"""
import tempfile
from pathlib import Path

import fitz  # PyMuPDF, installed with pymupdf4llm
import numpy as np
from PIL import Image, ImageFilter

from ingest import extract

TYPED = "Inspection Report QR-2207. Customer: Riverside Bus Co. Rejected: 18 pieces."
SCANNED = "Delivery Note DN-5531. Shipped to Hillview warehouse on 14 March 2025."


def scan_image(text, path):
    """A page with the text as pixels only: printed, slightly tilted, blurred and noisy, like a scan."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=300)
    page.insert_textbox(fitz.Rect(40, 40, 555, 260), text, fontsize=14, fontname="helv")
    pix = page.get_pixmap(dpi=200)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L")
    img = img.rotate(1.0, expand=True, fillcolor=255).filter(ImageFilter.GaussianBlur(0.8))
    a = np.array(img, dtype=np.float32) + np.random.default_rng(0).normal(0, 12, (img.height, img.width))
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(path)


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    scan_image(SCANNED, tmp / "scan.png")
    Image.open(tmp / "scan.png").convert("RGB").save(tmp / "photo.jpg")

    typed = fitz.open()  # a normal PDF: real text layer
    typed.new_page().insert_text((50, 80), TYPED, fontsize=11)
    typed.save(tmp / "typed.pdf")

    scanned = fitz.open()  # a scanned PDF: the page is only an image
    scanned.new_page(width=595, height=300).insert_image(fitz.Rect(0, 0, 595, 300), filename=str(tmp / "scan.png"))
    scanned.save(tmp / "scanned.pdf")

    mixed = fitz.open()  # page 1 typed, page 2 scanned
    mixed.new_page().insert_text((50, 80), TYPED, fontsize=11)
    mixed.new_page(width=595, height=300).insert_image(fitz.Rect(0, 0, 595, 300), filename=str(tmp / "scan.png"))
    mixed.save(tmp / "mixed.pdf")

    must = {"typed.pdf": ["QR-2207", "Riverside"], "scanned.pdf": ["DN-5531", "Hillview", "2025"],
            "mixed.pdf": ["QR-2207", "DN-5531", "Hillview"], "scan.png": ["DN-5531", "Hillview"],
            "photo.jpg": ["DN-5531", "Hillview"]}
    bad = 0
    for name, words in must.items():
        md = extract(tmp / name) or ""
        missing = [w for w in words if w.lower() not in md.lower()]
        bad += bool(missing)
        print(f"{'ok ' if not missing else 'BAD'} {name:12} {'' if not missing else f'missing {missing}: '}"
              f"{' '.join(md.split())[:110]!r}")
    print(f"FAILED {bad}" if bad else f"ALL OK {len(must)}")
