"""List Word's spelling and grammar findings for a DOCX (QA helper).

Usage: python spellcheck_word.py file.docx
"""
import os
import sys

import pythoncom
import win32com.client

src = os.path.abspath(sys.argv[1])
pythoncom.CoInitialize()
word = win32com.client.DispatchEx("Word.Application")
word.Visible = False
word.DisplayAlerts = 0
try:
    doc = word.Documents.Open(src, ReadOnly=True, AddToRecentFiles=False)
    sp = doc.SpellingErrors
    print("spelling errors:", sp.Count)
    seen = {}
    for i in range(1, sp.Count + 1):
        w = sp.Item(i).Text
        seen[w] = seen.get(w, 0) + 1
    for w, n in sorted(seen.items()):
        print(f"  {w!r} x{n}")
    gr = doc.GrammaticalErrors
    print("grammar findings:", gr.Count)
    for i in range(1, min(gr.Count, 60) + 1):
        try:
            r = gr.Item(i)
            ctx = r.Sentences(1).Text.strip().replace("\r", " ")[:140]
            print(f"  [{i}] {r.Text.strip()[:60]!r} :: {ctx}")
        except Exception as exc:  # noqa: BLE001
            print("  (could not read item", i, exc, ")")
    doc.Close(False)
finally:
    word.Quit()
