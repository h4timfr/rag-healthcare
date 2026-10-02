"""Render a DOCX to PDF with Microsoft Word (so pagination matches Word) and
export one PNG per page for visual inspection.

Usage: python render_pdf.py in.docx out.pdf [png_dir]
Requires: pywin32, PyMuPDF, Microsoft Word.
"""
import os
import sys

import pythoncom
import win32com.client

src = os.path.abspath(sys.argv[1])
dst = os.path.abspath(sys.argv[2])
png_dir = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else None

pythoncom.CoInitialize()
word = win32com.client.DispatchEx("Word.Application")
word.Visible = False
word.DisplayAlerts = 0
try:
    doc = word.Documents.Open(src, ReadOnly=True, AddToRecentFiles=False)
    doc.Repaginate()
    pages = doc.ComputeStatistics(2)
    doc.ExportAsFixedFormat(dst, 17)
    doc.Close(False)
    print("word pages:", pages)
finally:
    word.Quit()

import pymupdf

pdf = pymupdf.open(dst)
print("pdf pages:", pdf.page_count)
if png_dir:
    os.makedirs(png_dir, exist_ok=True)
    for i, page in enumerate(pdf):
        page.get_pixmap(dpi=110).save(os.path.join(png_dir, f"page{i + 1}.png"))
