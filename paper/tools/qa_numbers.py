"""QA: trace the paper's numbers to the frozen research artifacts.

1. Table I / II values are recomputed from research/results/counterbalanced_retrieval_*.json.
2. Table III values are recomputed from data/evaluation_summary.publication_run*.txt (local files).
3. Every other numeric token in the paper body is searched in the frozen research documents
   (FINAL_REPORT.md, RESULTS_TABLE.md, CONTRIBUTION.md, DEMO_SCRIPT.md, ARCHITECTURE.md, README.md)
   and listed if absent, for manual review.

Usage (repository root): python paper/tools/qa_numbers.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
tpl = (ROOT / "paper" / "tools" / "paper_template.md").read_text(encoding="utf-8")
frozen = json.loads((ROOT / "research/results/counterbalanced_retrieval_20261001T081145Z.json").read_text(encoding="utf-8"))
problems = []


def table_rows(caption_start):
    m = re.search(r"Table: " + re.escape(caption_start) + r".*?\n((?:\|.*\n?)+)", tpl)
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in m.group(1).strip().splitlines()]
    return [r for r in rows if not all(set(c) <= set("-:") for c in r)]


# ---- Table I
labels = {"Global dense": "A_global_dense", "Global hybrid": "A_global_hybrid", "Scoped dense": "B_scoped_dense",
          "Scoped hybrid": "B_scoped_hybrid", "Production": "C_production"}
for row in table_rows("Retrieval evidence metrics")[1:]:
    key = labels[row[0]]
    m = frozen["summary"]["means"][key]
    lat = frozen["summary"]["latency_ms"][key]
    want = [f"{m['recall']:.4f}", f"{m['mrr']:.4f}", f"{m['ndcg']:.4f}", f"{m['term_coverage']:.4f}",
            f"{lat['median']:.2f}", f"{lat['p95']:.2f}"]
    got = [c.rstrip("*") for c in row[1:]]
    if got != want:
        problems.append(("Table I", row[0], got, want))

# ---- Table II
cmpmap = {"G: H vs D": "A_global", "S: H vs D": "B_scoped", "P vs S-H": "C_vs_B"}
metmap = {"MRR@5": "mrr", "nDCG@5": "ndcg", "Recall@5": "recall", "Evid. cov.": "term_coverage"}
for row in table_rows("Paired Wilcoxon")[1:]:
    c = frozen["summary"]["comparisons"][f"{cmpmap[row[0]]}:{metmap[row[1]]}"]
    md = c["mean_diff"]
    want_md = f"{md:+.4f}" if abs(md) > 1e-12 else "0.0000"
    praw = c["p_value"]
    want = [want_md, str(c["wins_b"]), str(c["wins_a"]), str(c["ties"])]
    got = row[2:6]
    if got != want:
        problems.append(("Table II", row[:2], got, want))
    p_str = row[6]
    if abs(float(p_str) - praw) > 5e-4 * max(1, abs(praw)) and abs(float(p_str) - round(praw, len(p_str.split('.')[1]))) > 1e-9:
        problems.append(("Table II raw p", row[:2], p_str, praw))
    if c.get("p_holm") is not None and abs(float(row[7]) - c["p_holm"]) > 6e-4:
        problems.append(("Table II Holm p", row[:2], row[7], c["p_holm"]))

# ---- Table III
def summary(run):
    t = (ROOT / "data" / f"evaluation_summary.publication_run{run}.txt").read_text(encoding="utf-8")
    out = []
    for sec, nxt in (("DENSE SUMMARY", "HYBRID SUMMARY"), ("HYBRID SUMMARY", "COMPARISON")):
        blk = t[t.index(sec):t.index(nxt, t.index(sec) + 5)]
        g = lambda pat: re.search(pat, blk).group(1)
        out.append((g(r"Passes:\s+(\d+)"), g(r"Mean Term Coverage:\s+([\d.]+)"),
                    g(r"Retrieval failures:\s+(\d+)"), g(r"Generation failures:\s+(\d+)")))
    return out


for row in table_rows("Descriptive single-run")[1:]:
    run = row[0] if row[0] != "5" else "5_porter"
    (dp, dc, dr, dg), (hp, hc, hr, hg) = summary(run)
    want = [f"{dp} / {hp}", f"{dc} / {hc}", f"{dr} / {hr}", f"{dg} / {hg}"]
    if row[1:] != want:
        problems.append(("Table III", row[0], row[1:], want))

print("table checks:", "OK" if not problems else problems)

# ---- numeric tokens elsewhere
body = tpl.split("\n## I. Introduction")[1]
body = re.sub(r"Table:.*?\n(?:\|.*\n?)+", "", body)          # drop table blocks (checked above)
body = re.sub(r"\$\$.*?\$\$", "", body)                       # drop equations
body = re.sub(r"!\[.*?\]\(.*?\)(?:\{.*?\})?", "", body)       # drop figure markup
abstract = re.search(r"ABSTRACT: (.*)", tpl).group(1)
text = abstract + "\n" + body
docs = ""
for name in ("research/FINAL_REPORT.md", "research/RESULTS_TABLE.md", "research/CONTRIBUTION.md",
             "research/DEMO_SCRIPT.md", "research/ARCHITECTURE.md", "README.md"):
    docs += (ROOT / name).read_text(encoding="utf-8") + "\n"
docs_flat = docs.replace("–", "-")
tokens = sorted(set(re.findall(r"\d+(?:\.\d+)?%?", text)), key=lambda x: (len(x), x))
missing = [t for t in tokens if t not in docs_flat and t.rstrip("%") not in docs_flat]
print("numeric tokens checked:", len(tokens))
print("not found verbatim in the frozen documents (review manually):", missing)
