"""The generation fidelity tables, built from the statistics CSVs.

  ablation   the nineteen fidelity metrics for the five component ablations,
             the counterpart of table:ablation_portfolio_risk, which reports
             portfolio risk only
  benchmark  the same nineteen metrics against the baselines, which reproduces
             the upper half of table:statistical_fidelity_combined and is what
             checks this script against the numbers already in the paper

The metric set is the one the paper uses: the MAE variant of the three cross
correlation entries, then KS, Skew and Kurt at the four horizons, the three
autocorrelation entries and the leverage effect. EMD and the FROB and MSE
variants of the cross correlation entries are computed upstream but are not
reported, so they are dropped here rather than in the CSV.

Each table is written as a bare table environment so the paper can \\input it,
and needs booktabs and adjustbox, both already in the preamble.

Run from the repository root:
      python utils/make_paper_tables.py             both tables
      python utils/make_paper_tables.py ablation    only the ablation one
"""

import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(_HERE)
STATS = os.path.join(PROJECT, "result", "statistics")
OUT = os.path.join(PROJECT, "scratch")

# The CSV row key, the label the paper prints for it, and whether the group ends
# after it. Order is the order of the rows in the table, which follows the
# grouping of the combined table in the paper: dependence, marginals by horizon,
# the two shape moments, autocorrelation, leverage.
METRICS = [
    ("CCF(τ = 0)_MAE", r"CCF($\tau = 0$)", False),
    ("CCF(τ > 0)_MAE", r"CCF($\tau > 0$)", False),
    ("Vol CCF(τ = 0)_MAE", r"Vol CCF($\tau = 0$)", True),
    ("KS(1)", r"KS(1)", False),
    ("KS(5)", r"KS(5)", False),
    ("KS(20)", r"KS(20)", False),
    ("KS(100)", r"KS(100)", True),
    ("Skew(1)", r"Skew(1)", False),
    ("Skew(5)", r"Skew(5)", False),
    ("Skew(20)", r"Skew(20)", False),
    ("Skew(100)", r"Skew(100)", True),
    ("Kurt(1)", r"Kurt(1)", False),
    ("Kurt(5)", r"Kurt(5)", False),
    ("Kurt(20)", r"Kurt(20)", False),
    ("Kurt(100)", r"Kurt(100)", True),
    ("ACF(x)", r"ACF($x$)", False),
    ("ACF(|x|)", r"ACF($|x|$)", False),
    ("ACF(x^2)", r"ACF($x^{2}$)", True),
    ("Leverage Effect", r"Lev. Effect", False),
]

# The CSV column header and the column heading the paper prints for it. The
# ablation CSV calls the full model "Baseline"; the paper calls it "Full".
ABLATION_COLUMNS = [
    ("Baseline", r"\textbf{Full}"),
    ("w/o RV Decomp.", r"\textbf{w/o RV Decomp.}"),
    ("w/o LambertW", r"\textbf{w/o Lambert $W$}"),
    ("w/o DMS-TCN", r"\textbf{w/o DMS-TCN}"),
    ("w/o L1 loss", r"\textbf{w/o L1 loss}"),
    ("w/o cosine", r"\textbf{w/o cosine}"),
]

BENCHMARK_COLUMNS = [
    ("QuantDiffusion (ours, C2C/w10)", r"\textbf{QuantDiffusion}"),
    ("QuantGAN", r"\textbf{QuantGAN}"),
    ("TailGAN", r"\textbf{Tail-GAN}"),
    ("Diffusion-TS", r"\textbf{Diffusion-TS}"),
]


def read_stats(name):
    """Read one statistics CSV into {metric: {column: (mean, sd)}}.

    The files are written with a BOM and an unnamed first column, and every cell
    is the string "mean ± sd", so both numbers are parsed here and the rounding
    is left to the formatter: ranking on the rounded value would tie entries the
    full precision separates.
    """
    path = os.path.join(STATS, f"{name}__gen_quality.csv")
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    header = [c.strip() for c in rows[0]]
    table = {}
    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        cells = {}
        for col, cell in zip(header[1:], row[1:]):
            mean, sd = cell.split("±")
            cells[col] = (float(mean), float(sd))
        table[row[0].strip()] = cells
    return table


def format_row(label, cells, columns):
    """One metric as a LaTeX row, best in bold and second underlined.

    Every metric here is an error, so lower is better throughout. Ranking is on
    the mean alone; the standard deviation is reported but never breaks a tie.
    """
    means = [cells[key][0] for key, _ in columns]
    order = sorted(range(len(means)), key=lambda i: means[i])
    best, second = order[0], order[1]

    parts = [label]
    for i, (key, _) in enumerate(columns):
        mean, sd = cells[key]
        body = f"{mean:.4f}"
        if i == best:
            body = rf"\textbf{{{body}}}"
        elif i == second:
            body = rf"\underline{{{body}}}"
        parts.append(rf"& {body} $\pm$ {sd:.4f}")
    return "\n".join(parts) + r" \\"


def build(name, columns, caption, label):
    stats = read_stats(name)
    missing = [key for key, _, _ in METRICS if key not in stats]
    if missing:
        raise KeyError(f"{name}__gen_quality.csv is missing {missing}")

    body = []
    for i, (key, pretty, group_end) in enumerate(METRICS):
        body.append(format_row(pretty, stats[key], columns))
        if group_end and i != len(METRICS) - 1:
            body.append(r"\addlinespace")

    spec = "l" + "c" * len(columns)
    head = " & ".join([r"\textbf{Metric}"] + [h for _, h in columns]) + r" \\"

    return "\n".join(
        [
            "% Generated by utils/make_paper_tables.py from",
            f"% result/statistics/{name}__gen_quality.csv. Do not edit by hand.",
            r"\begin{table}[t]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\renewcommand{\arraystretch}{0.9}",
            r"\begin{adjustbox}{max width=\columnwidth}",
            rf"\begin{{tabular}}{{{spec}}}",
            r"\toprule",
            head,
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\end{table}",
            "",
        ]
    )


def ablation():
    tex = build(
        "ablation",
        ABLATION_COLUMNS,
        "Ablation of the QuantDiffusion Components, Generation Fidelity. "
        "Lower is better, \\textbf{bold} is best and \\underline{underline} second.",
        "table:ablation_generation_fidelity",
    )
    dst = os.path.join(OUT, "ablation_gen_fidelity.tex")
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(tex)
    print(f"  wrote {os.path.relpath(dst, PROJECT)}")


def benchmark():
    tex = build(
        "benchmark",
        BENCHMARK_COLUMNS,
        "Comparison Against Baselines, Generation Fidelity. "
        "Lower is better, \\textbf{bold} is best and \\underline{underline} second.",
        "table:benchmark_generation_fidelity",
    )
    dst = os.path.join(OUT, "benchmark_gen_fidelity.tex")
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(tex)
    print(f"  wrote {os.path.relpath(dst, PROJECT)}")


TABLES = {"ablation": ablation, "benchmark": benchmark}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(TABLES)
    for key in wanted:
        if key not in TABLES:
            raise SystemExit(f"unknown table {key!r}, pick from {list(TABLES)}")
        TABLES[key]()
