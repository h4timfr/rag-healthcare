"""Generate the paper's figures and table images.

Every number drawn here is copied from the frozen research package:
- Table I / Fig. 3: research/results/counterbalanced_retrieval_20261001T081145Z.json
  (also research/FINAL_REPORT.md Section 11);
- Table II: the same file's paired comparisons (FINAL_REPORT.md Section 12);
- Table III: FINAL_REPORT.md Section 13.1 (runs 1-5, research harness);
- Fig. 5: the 2026-10-02 demo check recorded in FINAL_REPORT.md Section 14
  and research/DEMO_SCRIPT.md segment 6.

Usage (from the repository root): python paper/tools/make_figures.py
Requires matplotlib.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
TAB = ROOT / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 7.5,
    "axes.linewidth": 0.6,
    "savefig.dpi": 300,
    "mathtext.fontset": "stix",
    "hatch.linewidth": 0.5,
})

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"
BOX = "#f3f2ee"
EDGE = "#3a3936"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical slots 1-3 (all-pairs pass)
HILITE = "#fde7dd"
COL_W = 3.45  # single IEEE column, inches


def box(ax, x, y, w, h, text, fc=BOX, ec=EDGE, lw=0.7, ls="-", size=7.0, weight="normal", style="normal"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=INK, weight=weight, style=style,
            linespacing=1.15)


def arrow(ax, x1, y1, x2, y2, ls="-", color=EDGE):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, lw=0.7,
                                 color=color, ls=ls, shrinkA=0, shrinkB=0))


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax



class Inch:
    """Inch-based layout helper: y is measured downward from the top edge."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.fig, self.ax = canvas(w, h)

    def fx(self, x):
        return x / self.w

    def fy(self, y):
        return 1 - y / self.h

    def box(self, x, y, w, h, text, **kw):
        box(self.ax, self.fx(x), self.fy(y), w / self.w, h / self.h, text, **kw)

    def arrow(self, x1, y1, x2, y2, **kw):
        arrow(self.ax, self.fx(x1), self.fy(y1), self.fx(x2), self.fy(y2), **kw)

    def line(self, xs, ys, **kw):
        self.ax.plot([self.fx(x) for x in xs], [self.fy(y) for y in ys], **kw)

    def text(self, x, y, t, **kw):
        self.ax.text(self.fx(x), self.fy(y), t, **kw)

    def save(self, path):
        self.fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
        plt.close(self.fig)

# --------------------------------------------------------------------------- Fig. 1
def fig1():
    f = Inch(COL_W, 3.28)
    W = COL_W
    # ingestion row
    ing = [(0.26, 0.46, "PDF\nupload"), (1.00, 0.78, "Extraction\nPyMuPDF"),
           (1.97, 0.92, "Chunking\n(200 words,\n40 overlap)"), (2.975, 0.85, "Embedding +\nindexes: FAISS,\nSQLite, FTS5")]
    for x, w, t in ing:
        f.box(x, 0.19, w, 0.34, t, size=6.3)
    for (x1, w1, _), (x2, w2, _) in zip(ing, ing[1:]):
        f.arrow(x1 + w1 / 2, 0.19, x2 - w2 / 2, 0.19)
    f.line([0.05, W - 0.05], [0.47, 0.47], color=GRID, lw=0.7)
    f.text(W / 2, 0.47, " per question ", ha="center", va="center", fontsize=6.2, color=INK2, style="italic",
           bbox=dict(fc="white", ec="none", pad=0.6))
    f.box(W / 2, 0.62, 1.10, 0.17, "User question", fc="#ffffff", size=6.6, style="italic")
    f.box(0.92, 1.01, 1.38, 0.38, "Dense retrieval\nFAISS IndexFlatIP\n(cosine), top 10", size=6.4)
    f.box(2.48, 1.01, 1.38, 0.38, "Lexical retrieval\nSQLite FTS5, BM25\n(Porter), top 10", size=6.4)
    f.arrow(1.45, 0.705, 1.15, 0.82)
    f.arrow(2.04, 0.705, 2.34, 0.82)
    rows = [
        (1.50, "Weighted Reciprocal Rank Fusion\n(dense 2.0, lexical 1.0, k = 60)"),
        (1.88, "Document scoping and adaptive retrieval\n(top-k 5; 8 for explicit lists; per part for multi-part)"),
        (2.26, "Context assembly: evidence order, intent-specific\nanswer style, personal-advice reminder"),
        (2.64, "Local generation: Ollama, llama3.2:3b\n(temperature 0.1, 256/768-token budget)"),
        (3.02, "Answer + sources + verbatim evidence + timing\n(Streamlit interface)"),
    ]
    hh = 0.31
    f.arrow(0.92, 1.20, 1.30, rows[0][0] - hh / 2)
    f.arrow(2.48, 1.20, 2.10, rows[0][0] - hh / 2)
    for y, t in rows:
        f.box(W / 2, y, 3.2, hh, t, size=6.4)
    for (y1, _), (y2, _) in zip(rows, rows[1:]):
        f.arrow(W / 2, y1 + hh / 2, W / 2, y2 - hh / 2)
    f.save(FIG / "fig1_architecture.png")


# --------------------------------------------------------------------------- Fig. 2
def fig2():
    f = Inch(COL_W, 2.3)
    f.text(0.86, 0.07, "Retrieval evaluation (Tier 1)", ha="center", va="center", fontsize=6.9, weight="bold")
    f.text(2.59, 0.07, "Generation evaluation (Tier 2)", ha="center", va="center", fontsize=6.9, weight="bold")
    f.line([COL_W / 2, COL_W / 2], [0.02, 2.25], color=GRID, lw=0.7)
    left = ["Research corpus\n5 PDFs, 225 chunks", "45 benchmark questions\n(page + term labels)",
            "5 retrieval configurations\n(global/scoped x dense/hybrid,\nproduction path)",
            "Recall@5, MRR@5, nDCG@5,\nevidence term coverage", "Wilcoxon signed-rank +\nHolm correction (10 tests)"]
    right = ["End-to-end runs 1-5\n(whole-corpus, neutral prompt)", "Answer coverage +\njoint pass criterion",
             "Failure taxonomy\n(Tier 3 interpretation)", "Exploratory intervention\nstudy (one document)",
             "Safety probes\n(10 probes x 3 repeats)"]
    ys = [0.40, 0.78, 1.20, 1.66, 2.04]
    hs = [0.27, 0.27, 0.36, 0.27, 0.27]
    for x, items in ((0.86, left), (2.59, right)):
        for y, h, t in zip(ys, hs, items):
            f.box(x, y, 1.50, h, t, size=6.3)
        for i in range(len(ys) - 1):
            f.arrow(x, ys[i] + hs[i] / 2, x, ys[i + 1] - hs[i + 1] / 2)
    f.text(COL_W / 2, 0.59, "same 45\nquestions", ha="center", va="center", fontsize=5.8, color=INK2,
           bbox=dict(fc="white", ec="none", pad=0.6))
    f.save(FIG / "fig2_evaluation_pipeline.png")


# --------------------------------------------------------------------------- Fig. 3
CONFIGS = ["Global\ndense", "Global\nhybrid", "Scoped\ndense", "Scoped\nhybrid", "Production\npath"]
RECALL = [0.9111, 0.9333, 0.9333, 0.9333, 0.9333]
MRR = [0.6922, 0.8093, 0.7396, 0.8259, 0.8296]
NDCG = [0.7463, 0.8403, 0.7874, 0.8530, 0.8561]


def fig3():
    fig, ax = plt.subplots(figsize=(COL_W, 2.05))
    fig.subplots_adjust(left=0.105, right=0.995, top=0.86, bottom=0.17)
    n = len(CONFIGS)
    bw = 0.25
    hatches = ["", "////", "...."]
    for j, (label, vals) in enumerate((("Recall@5", RECALL), ("MRR@5", MRR), ("nDCG@5", NDCG))):
        xs = [i + (j - 1) * (bw + 0.02) for i in range(n)]
        bars = ax.bar(xs, vals, width=bw, color=SERIES[j], edgecolor="white", linewidth=0.6,
                      hatch=hatches[j], label=label, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=5.2, color=INK, rotation=90)
    ax.set_xticks(range(n))
    ax.set_xticklabels(CONFIGS, fontsize=6.6, color=INK)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", labelsize=6.6, colors=INK2, length=2)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Score (n = 45 questions)", fontsize=6.8, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2), frameon=False, fontsize=6.8,
              handlelength=1.6, columnspacing=1.2)
    fig.savefig(FIG / "fig3_retrieval_results.png", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# --------------------------------------------------------------------------- Fig. 4
def fig4():
    f = Inch(COL_W, 2.2)
    f.box(COL_W / 2, 0.12, 3.35, 0.20, "Observed failure modes (qualitative taxonomy, Tier 3)", size=6.8,
          weight="bold", fc="#ffffff")
    bands = [
        (0.43, 0.36, "Retrieval\nstage",
         "F1 Retrieval miss: expected page not in top 5\nF2 Wrong-context / wrong-period evidence used"),
        (0.99, 0.70, "Generation\nstage\n(evidence\nretrieved)",
         "F3 False absence (\"not mentioned\")\nF4 Entity / value confusion\nF5 Component omission (multi-part)\n"
         "F6 Section / list confusion\nF7 Table-row / cell confusion"),
        (1.58, 0.36, "Presentation\nand safety", "F8 Documented treatment restated as a\nrecommendation without attribution"),
    ]
    for yc, h, label, body in bands:
        f.box(0.42, yc, 0.76, h, label, size=6.2, weight="bold", fc="#e9e8e3")
        f.box(2.06, yc, 2.52, h, body, size=6.3)
    f.text(COL_W / 2, 2.00, "Contributing upstream factors (inferred): two-column PDF extraction,\n"
           "heading-less overlap chunks, lexical grading of paraphrases",
           ha="center", va="center", fontsize=6.0, color=INK2, style="italic")
    f.save(FIG / "fig4_failure_taxonomy.png")


# --------------------------------------------------------------------------- Fig. 5
def fig5():
    f = Inch(COL_W, 2.05)
    f.box(COL_W / 2, 0.17, 3.35, 0.30,
          "Query: age-adjusted prevalence of diagnosed diabetes, 2017-2018,\n"
          "adults with less than a high school education (public CDC report)", fc="#ffffff", size=6.5, style="italic")
    f.text(0.88, 0.45, "Retrieved top 5 (PDF pages)", ha="center", va="center", fontsize=6.6, weight="bold")
    f.text(2.58, 0.45, "Not retrieved", ha="center", va="center", fontsize=6.6, weight="bold")
    pages = [("p. 19", "Appendix Table 4 (incidence)", 0.17), ("p. 7", "county-level prevalence", 0.17),
             ("p. 17", "Appendix Table 1, 2013-2016\nless than HS: 12.7 (11.4-14.2)", 0.30),
             ("p. 28", "incidence trends", 0.17), ("p. 8", "incidence trends (notes)", 0.17)]
    y = 0.58
    y17 = None
    for pg, desc, h in pages:
        hi = pg == "p. 17"
        yc = y + h / 2
        f.box(0.85, yc, 1.55, h, f"{pg}  {desc}", size=6.0, fc=HILITE if hi else BOX,
              ec=SERIES[1] if hi else EDGE, lw=1.0 if hi else 0.6)
        if hi:
            y17 = yc
        y += h + 0.04
    f.box(2.60, 0.86, 1.55, 0.52, "p. 18  Appendix Table 3, 2017-2018\nless than HS: Total 13.3,\nMen 13.0, Women 13.6",
          size=6.0, ls=(0, (3, 2)), fc="#ffffff")
    f.text(2.58, 1.18, "correct table missed", ha="center", va="center", fontsize=5.8, color=INK2, style="italic")
    f.box(2.60, 1.58, 1.55, 0.50, "Answer in 3 of 3 runs:\n\"12.7 (11.4-14.2)\"\n(2013-2016 value, period not stated)",
          size=6.0, fc=HILITE, ec=SERIES[1], lw=1.0)
    f.line([1.625, 1.73, 1.73], [y17, y17, 1.58], color=SERIES[1], lw=0.7)
    f.arrow(1.73, 1.58, 1.82, 1.58, color=SERIES[1])
    f.save(FIG / "fig5_demo_failure.png")


# --------------------------------------------------------------------------- table images
def table_png(name, header, rows, widths, title, height):
    fig, ax = plt.subplots(figsize=(7.6, height))
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=header, colWidths=widths, loc="upper center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(6.8)
    t.scale(1, 1.25)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor(BOX)
    ax.set_title(title, fontsize=8, pad=6)
    fig.savefig(TAB / name, bbox_inches="tight")
    plt.close(fig)


def tables():
    lat = [("26.01", "28.59"), ("28.23", "30.45"), ("28.33", "30.85"), ("31.32", "34.23"), ("40.79*", "53.53*")]
    rows = [[c.replace("\n", " "), f"{r:.4f}", f"{m:.4f}", f"{n:.4f}", f"{r:.4f}", a, b]
            for c, r, m, n, (a, b) in zip(CONFIGS, RECALL, MRR, NDCG, lat)]
    table_png("table1_retrieval_results.png",
              ["Configuration", "Recall@5", "MRR@5", "nDCG@5", "Evidence term cov.", "Median (ms)", "P95 (ms)"],
              rows, [0.2, 0.11, 0.11, 0.11, 0.17, 0.12, 0.11],
              "TABLE I. Retrieval evidence metrics and latency (n = 45; * includes prompt construction)", 1.4)
    stats = [
        ["Global hybrid vs dense", "MRR@5", "+0.1170", "13", "0", "32", "0.000244", "0.00244", "significant"],
        ["Global hybrid vs dense", "nDCG@5", "+0.0941", "13", "0", "32", "0.000244", "0.00244", "significant"],
        ["Global hybrid vs dense", "Recall@5", "+0.0222", "1", "0", "44", "1.000", "1.000", "n.s."],
        ["Global hybrid vs dense", "Evid. term cov.", "+0.0222", "1", "0", "44", "1.000", "1.000", "n.s."],
        ["Scoped hybrid vs dense", "MRR@5", "+0.0863", "11", "2", "32", "0.0210", "0.168", "n.s. (directional)"],
        ["Scoped hybrid vs dense", "nDCG@5", "+0.0656", "11", "2", "32", "0.0256", "0.179", "n.s. (directional)"],
        ["Scoped hybrid vs dense", "Recall@5", "0.0000", "1", "1", "43", "1.000", "1.000", "n.s."],
        ["Scoped hybrid vs dense", "Evid. term cov.", "0.0000", "1", "1", "43", "1.000", "1.000", "n.s."],
        ["Production vs scoped hybrid", "MRR@5", "+0.0037", "2", "0", "43", "0.500", "1.000", "n.s."],
        ["Production vs scoped hybrid", "nDCG@5", "+0.0031", "2", "0", "43", "0.500", "1.000", "n.s."],
    ]
    table_png("table2_statistical_results.png",
              ["Comparison", "Metric", "Mean diff.", "Wins", "Losses", "Ties", "Raw p", "Holm p", "Interpretation"],
              stats, [0.28, 0.13, 0.08, 0.05, 0.06, 0.05, 0.09, 0.08, 0.17],
              "TABLE II. Paired Wilcoxon signed-rank tests with Holm correction (n = 45)", 2.3)
    e2e = [["1", "34 / 35", "0.822 / 0.856", "4 / 3", "7 / 7"], ["2", "35 / 34", "0.844 / 0.833", "4 / 3", "6 / 8"],
           ["3", "34 / 32", "0.822 / 0.800", "4 / 3", "7 / 10"], ["4", "36 / 35", "0.856 / 0.856", "4 / 3", "5 / 7"],
           ["5", "35 / 32", "0.844 / 0.800", "4 / 3", "6 / 10"]]
    table_png("table3_end_to_end_results.png",
              ["Run", "Joint pass (D / H)", "Answer term cov. (D / H)", "Retrieval failures (D / H)",
               "Generation failures (D / H)"],
              e2e, [0.1, 0.2, 0.24, 0.23, 0.23],
              "TABLE III. Descriptive single-run end-to-end evaluations (whole-corpus retrieval, neutral prompt, "
              "256 tokens; not the production path)", 1.45)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    tables()
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))
    print("tables:", sorted(p.name for p in TAB.glob("*.png")))
