"""The five figures that go in the paper.

  fig1  stylized facts:  ACF(|r|), ACF(r^2), cross asset comovement of x_t
  fig2  five day cumulative log return distributions for JPM, the figure already
        in main.tex, restyled
  fig3  portfolio risk radars, benchmark beside ablation, same axes and format
  fig4  risk error at every horizon, the Tail-GAN metric by h
  fig5  architecture, a temporary raster stand in for the draw.io figure

Everything is sized for acmart sigconf: 7.0 in across both columns, 3.33 in for
one, Times New Roman to match the body text, 400 dpi.

The figures are written straight into result/images.

Run from the repository root:
      python utils/make_paper_figures.py          all five
      python utils/make_paper_figures.py 1 4      only 1 and 4
or from a notebook: import utils.make_paper_figures as pf; pf.fig1()

fig4 needs TailGAN's Compute_PNL, which segfaults on CPU at interpreter teardown
under load. Each horizon is therefore computed in a subprocess that saves its
P&L and leaves through os._exit, and the result is cached under pnl_cache/, which
is safe to delete. Set CUDA_VISIBLE_DEVICES=-1 to keep it off a busy GPU.
"""

import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import MaxNLocator

# This module lives in utils/ while the plotting helpers it draws on live beside
# the older figure candidates, so both paths are resolved from this file rather
# than from the working directory: the notebook, a shell in the repository root
# and the pnl worker subprocess all import it the same way.
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(_HERE)
HELPERS = os.path.join(PROJECT, "result", "images", "figures")
if HELPERS not in sys.path:
    sys.path.insert(0, HELPERS)

from make_figures import (ABL, ASSETS, C, DPI, EPS, MODELS, ROOT,  # noqa: E402
                          RV_WINDOW, ST, acf_batch, flat_windows, load_model,
                          load_real, read_pm_table, tag)
from make_square_figures import COL_ABL, COL_BENCH, RISK, radar  # noqa: E402

# the figures land directly under result/images
OUT = os.path.join(PROJECT, "result", "images")
# Inside the helpers directory, which .gitignore covers, because these 6.7 MB of
# intermediate arrays are a side effect of the subprocess workaround below and
# have no business being tracked next to the figures they produce.
CACHE = os.path.join(HELPERS, "pnl_cache")
os.makedirs(OUT, exist_ok=True)
FULL, COL = 7.0, 3.33          # acmart sigconf: both columns, one column
HS = [20, 40, 64, 96, 128, 160, 200, 255]
ALPHA = 0.05
N_WIN = 400                    # scenarios per model where a subset suffices

# Times New Roman everywhere, with a serif math face to match it
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.6,
    "axes.labelsize": 7.6,
    "axes.titlesize": 8.2,
    "legend.fontsize": 6.8,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
})


def save(fig, name, outdir=OUT):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, name)
    fig.savefig(p, dpi=DPI, facecolor="white")
    plt.close(fig)
    print("  wrote", os.path.relpath(p, ROOT))


# the one order every figure uses, matching the columns of the result tables
ORDER = ["QuantGAN", "TailGAN", "Diffusion-TS", "QuantDiffusion"]
# the keys are the folder names; the paper spells the model Tail-GAN
DISPLAY = {"TailGAN": "Tail-GAN"}
SHORT = {"QuantGAN": "QuantGAN", "TailGAN": "Tail-GAN",
         "Diffusion-TS": "Diffusion-TS", "QuantDiffusion": "ours"}


def relabel(ax, **kw):
    """Draw the legend with the paper's spelling of each model."""
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, [DISPLAY.get(x, x) for x in l], **kw)


def order_models():
    return [m for m in ORDER if m in MODELS]


# ==========================================================================
# fig 1: stylized facts, three panels in a row
# ==========================================================================
def log_rv(A, w=RV_WINDOW):
    """x_t = log RV_t per asset, the second half of the diffusion representation."""
    c = np.cumsum(A ** 2, axis=-1)
    rv = (c[..., w:] - c[..., :-w]) / w
    return np.log(rv + EPS)


def comovement_windows(A):
    """Mean pairwise correlation of x_t across assets, one value per scenario.

    Volatility moves together across markets, and a model can reproduce each
    asset's own clustering while getting the joint behaviour wrong, so this is
    what separates a per series fit from a joint one.

    The mean of this vector is the Vol Comovement entry of the fidelity table,
    0.39 for the real test set. Panel (c) keeps the vector instead of the mean
    because the two are far from equivalent here: a real 256 day scenario ranges
    from 0.07 in the quietest stretch to 0.75 in the one containing the 2020
    crash, and it is the upper end that removes diversification from a portfolio.

    Every scenario is used, with no subsetting. The level moves a long way with
    the window set, 0.39 over all 1,538 real test windows against 0.54 over the
    first 300, so truncating here would put this panel at odds with the table.
    """
    x = log_rv(np.asarray(A, dtype=np.float64))
    xc = x - x.mean(axis=2, keepdims=True)
    sd = x.std(axis=2) + EPS
    cov = np.einsum("nit,njt->nij", xc, xc) / x.shape[2]
    corr = cov / (sd[:, :, None] * sd[:, None, :])
    iu = np.triu_indices(x.shape[1], 1)
    return np.nanmean(corr[:, iu[0], iu[1]], axis=1)


def fig1(panel_c="tail", name="fig1_stylized_facts.png", outdir=OUT):
    """Panel (c) has four candidate forms, all of the same per scenario vector.

    tail      the one in use. Comovement on the vertical axis against how rare
              the scenario is on the horizontal, so panel (c) reads in the same
              idiom as (a) and (b), statistic vertical and swept variable
              horizontal, and the right hand edge is the crisis end of the
              market where each model's ceiling can be read off.
    survival  the same curve untransposed, comovement horizontal and the
              fraction of scenarios above it on a log vertical axis.
    cdf       the distribution as it is, which spends the panel on the bulk and
              compresses the crisis end against the top of the axis.
    bars      the mean, as before, with the 95th percentile added as a marker,
              so the two summary numbers are visible and the shape is not.
    """
    nlags = 40
    real, _ = load_real()
    series = {"Real": real}
    for m in order_models():
        a = load_model(MODELS[m])
        if a is not None:
            series[m] = a

    # three panels in a row inside one column, so each gets about an inch. What
    # makes that fit: (a) and (b) share the y axis, the lag axis carries three
    # ticks, and (c) carries no category labels because every curve is coloured
    # by model and the legend above already maps colour to model.
    fig, axes = plt.subplots(1, 3, figsize=(COL, 1.62), sharey=False,
                             gridspec_kw={"width_ratios": [1, 1, 0.86]})
    fig.subplots_adjust(left=0.125, right=0.99, top=0.755, bottom=0.235,
                        wspace=0.30)

    panels = [
        (r"ACF$(|r_\tau|)$", lambda A: acf_batch(np.abs(flat_windows(A, N_WIN)),
                                                 nlags).mean(0)),
        (r"ACF$(r_\tau^2)$", lambda A: acf_batch(flat_windows(A, N_WIN) ** 2,
                                                 nlags).mean(0)),
    ]
    for ax, (title, fn), letter in zip(axes, panels, "ab"):
        for k, A in series.items():
            ax.plot(np.arange(1, nlags + 1), fn(A), color=C[k],
                    lw=1.1 if k == "Real" else 0.75,
                    ls="-" if k == "Real" else "--",
                    label=k if letter == "a" else None, zorder=3)
        ax.set_title(title, pad=2.0, fontsize=6.4)
        ax.set_xlabel(r"lag $\tau$", fontsize=5.8, labelpad=0.8)
        ax.set_xticks([0, 20, 40])
        ax.set_yticks([0.0, 0.1, 0.2])
        ax.axhline(0, color="#999999", lw=0.5, zorder=1)
        ax.tick_params(labelsize=5.4, pad=1.2)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        tag(ax, letter, dx=-0.30, dy=0.99)
    axes[0].set_ylabel("autocorrelation", fontsize=5.8, labelpad=1.2)

    ax = axes[2]
    # The five generated sets are pooled, which leaves the mean of each curve
    # equal to the Vol Comovement entry of the fidelity table while letting the
    # spread be read off.
    m = {}
    for k, A in series.items():
        m[k] = comovement_windows(A) if k == "Real" else np.concatenate(
            [comovement_windows(load_model(MODELS[k], r)) for r in range(5)
             if load_model(MODELS[k], r) is not None])

    if panel_c in ("cdf", "survival", "tail"):
        # drawn through a few hundred quantiles rather than all 10,000 points,
        # which is the same curve at a fraction of the file size
        grid = np.linspace(0.0, 1.0 if panel_c == "cdf" else 0.99, 400)
        for k in series:
            q = np.quantile(m[k], grid)
            xy = (q, grid) if panel_c == "cdf" else \
                 (q, 1.0 - grid) if panel_c == "survival" else (1.0 - grid, q)
            ax.plot(*xy, color=C[k], lw=1.1 if k == "Real" else 0.75,
                    ls="-" if k == "Real" else "--", zorder=3)
        if panel_c == "cdf":
            ax.set_xlim(-0.12, 0.88)
            ax.set_xticks([0.0, 0.4, 0.8])
            ax.set_ylim(0, 1)
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.set_xlabel("comovement", fontsize=5.8, labelpad=0.8)
            ax.set_ylabel("fraction", fontsize=5.8, labelpad=1.2)
        elif panel_c == "survival":
            ax.set_xlim(-0.12, 0.88)
            ax.set_xticks([0.0, 0.4, 0.8])
            ax.set_yscale("log")
            ax.set_ylim(1e-2, 1)
            ax.set_yticks([1e-2, 1e-1, 1])
            ax.set_yticklabels(["1%", "10%", "100%"])
            ax.set_xlabel("comovement", fontsize=5.8, labelpad=0.8)
            ax.set_ylabel("scenarios above", fontsize=5.8, labelpad=1.2)
        else:
            # rarity increases to the right, so the crisis end of the market is
            # the right hand edge and the curve rises into it
            ax.set_xscale("log")
            ax.set_xlim(1.0, 1e-2)
            ax.set_xticks([1, 1e-1, 1e-2])
            ax.set_xticklabels(["100%", "10%", "1%"])
            ax.set_ylim(-0.12, 0.88)
            ax.set_yticks([0.0, 0.4, 0.8])
            ax.set_xlabel("scenarios above", fontsize=5.8, labelpad=0.8)
            ax.set_ylabel("comovement", fontsize=5.8, labelpad=1.2)
            # names the channel being correlated, since x_t is defined far from
            # this panel and the reader should not have to hunt for it
            ax.text(0.97, 0.03, r"$x_t = \log RV_t$", transform=ax.transAxes,
                    fontsize=5.0, va="bottom", ha="right", color="#444444")
    else:
        ms = [k for k in series if k != "Real"]
        ax.bar(range(len(ms)), [m[k].mean() for k in ms], 0.66,
               color=[C[k] for k in ms], lw=0, zorder=3)
        ax.plot(range(len(ms)), [np.quantile(m[k], 0.95) for k in ms], "D",
                ms=2.2, mfc="white", mew=0.7, mec=C["Real"], ls="none", zorder=5)
        rm, rq = m["Real"].mean(), np.quantile(m["Real"], 0.95)
        ax.axhline(rm, lw=0.9, color=C["Real"], zorder=4)
        ax.axhline(rq, lw=0.9, ls=(0, (2, 1.4)), color=C["Real"], zorder=4)
        for y, t in [(rm, "mean"), (rq, "q95")]:
            ax.text(len(ms) - 0.45, y, t, fontsize=4.4, va="bottom", ha="right",
                    color=C["Real"])
        ax.set_xticks([])                 # the bar colours carry the identity
        ax.set_xlim(-0.7, len(ms) - 0.3)
        ax.set_ylim(0, 0.88)
        ax.set_yticks([0.0, 0.4, 0.8])
        ax.set_xlabel("comovement", fontsize=5.8, labelpad=0.8)
    ax.set_title(r"Corr$(x_i, x_j)$", pad=2.0, fontsize=6.4)
    ax.tick_params(labelsize=5.4, pad=1.2)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    tag(ax, "c", dx=-0.30, dy=0.99)

    # one boxed row above the panels, as in fig4. It is attached to the figure
    # rather than to axes[0], which is only a third of the width here, so that it
    # centres over all three panels.
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, [DISPLAY.get(x, x) for x in l], ncol=5, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), frameon=True, fancybox=False,
               edgecolor="#8A8A8A", facecolor="white", framealpha=1.0,
               fontsize=4.8, handlelength=1.0, handletextpad=0.3,
               columnspacing=0.6, borderpad=0.28, borderaxespad=0.0)
    save(fig, name, outdir)


# ==========================================================================
# fig 2: the distribution figure already in the paper
# ==========================================================================
def cum_h(A, asset, h=5, n_win=800):
    """Non overlapping h step sums for one asset, pooled over scenarios."""
    A = np.asarray(A[:n_win, asset, :], dtype=np.float64)
    m = (A.shape[-1] // h) * h
    return A[:, :m].reshape(A.shape[0], -1, h).sum(-1).ravel()


# slate against terracotta: two low saturation, mid dark tones that hold their
# identity where they overlap, separate under a greyscale print, and read as
# print colours rather than as screen defaults
HIST_FC, HIST_EC = "#3F5D77", "#2B4155"
SYN_FC, SYN_EC = "#BE7250", "#8F5133"


def fig2(asset=2, h=5):
    """The figure already in main.tex, same layout, quieter colours.

    Overlapping filled histograms and a per panel y axis, exactly as the original,
    since that is what the caption in the paper describes. Only the two fills
    changed.
    """
    real, _ = load_real()
    hist = cum_h(real, asset, h, n_win=len(real))
    order = [m for m in order_models() if load_model(MODELS[m]) is not None]
    bins = np.linspace(*np.percentile(hist, [0.05, 99.95]), 61)

    # two by two so the four panels fit one column instead of the text width
    fig, axes = plt.subplots(2, 2, figsize=(COL, 2.95), sharex=True)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.915, bottom=0.135,
                        wspace=0.30, hspace=0.40)
    flat = axes.ravel()
    for k, (ax, m) in enumerate(zip(flat, order)):
        syn = cum_h(load_model(MODELS[m]), asset, h)
        ax.hist(hist, bins=bins, density=True, color=HIST_FC, alpha=0.62,
                edgecolor=HIST_EC, lw=0.2, label="Historical", zorder=2)
        ax.hist(syn, bins=bins, density=True, color=SYN_FC, alpha=0.62,
                edgecolor=SYN_EC, lw=0.2, label="Synthetic", zorder=3)
        ax.set_title(DISPLAY.get(m, m), pad=2.5, fontsize=7.0,
                     fontweight="bold")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(labelsize=5.8)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        if k % 2 == 0:
            ax.set_ylabel("Density", fontsize=6.6, labelpad=1.5)
        if k >= 2:
            ax.set_xlabel(f"Cumulative log return ({h} Day)", fontsize=6.6,
                          labelpad=1.5)
    flat[0].legend(loc="upper left", fontsize=5.4, handlelength=0.9,
                   labelspacing=0.18, borderpad=0.18, handletextpad=0.4)
    save(fig, "fig2_distribution.png")


# ==========================================================================
# fig 3: the two risk radars, benchmark beside ablation
# ==========================================================================
def fig3():
    bm, _ = read_pm_table(os.path.join(ST, "benchmark__tailgan_vares.csv"))
    ab, _ = read_pm_table(os.path.join(ST, "ablation__tailgan_vares.csv"))

    # stacked, one column wide. The twenty four printed values per panel set the
    # floor on how small each radar can get, since point sizes do not shrink with
    # the figure, so the pair goes vertically rather than side by side.
    fig = plt.figure(figsize=(COL, 4.55))
    for idx, (mean, cols, keys, title, letter) in enumerate([
            (bm, COL_BENCH, ORDER, "benchmark", "a"),
            (ab, COL_ABL, list(COL_ABL), "ablation", "b")]):
        present = {k: cols[k] for k in keys
                   if k in cols and cols[k] in mean.columns}
        vals = {m: {k: mean.loc[k, c] for k, _ in RISK}
                for m, c in present.items()}
        ax = fig.add_subplot(2, 1, idx + 1, projection="polar")
        radar(ax, RISK, vals, list(present), fmt="{:.3f}", note=False)
        ax.set_ylim(0, 1.17)           # tighter than the default outer margin
        for t in ax.get_xticklabels():
            t.set_fontsize(6.0)
        relabel(ax, loc="upper left", bbox_to_anchor=(-0.24, 1.12),
                fontsize=5.6, handlelength=0.9, labelspacing=0.2,
                handletextpad=0.4)
        ax.text(-0.26, 1.10, f"({letter})", transform=ax.transAxes,
                fontsize=7.6, fontweight="bold", va="bottom", ha="left")
        ax.text(1.06, 1.10, title, transform=ax.transAxes, fontsize=7.4,
                va="bottom", ha="right")
    # no footnote inside the figure: "further from the centre is better" belongs
    # in the caption, and at this size it would land on the ES_0.01 labels
    fig.subplots_adjust(left=0.10, right=0.90, top=0.925, bottom=0.025,
                        hspace=0.34)
    save(fig, "fig3_risk_radars.png")


# ==========================================================================
# fig 4: risk error at every horizon
# ==========================================================================
def _cache_path(h, k):
    return os.path.join(CACHE, f"h{h}_{k}.npy")


def pnl_worker(h):
    """Compute_PNL at one horizon, saving each series the moment it is done.

    TailGAN's CPU path corrupts the heap somewhere inside Compute_PNL. The P&L
    values come out correct, then the process dies with an access violation at
    whatever allocates next, which is why the crash moved between horizons and
    once landed inside np.savez_compressed with every number already computed.
    There is no way to catch that, so nothing is kept in memory: each series is
    written as its own uncompressed .npy as soon as it exists, the worker leaves
    through os._exit to skip teardown, and a rerun resumes from what survived.
    """
    sys.path.insert(0, os.path.join(ROOT, "modules", "TailGAN"))
    import tailgan_eval as TE

    TG = TE.configure_from_project(ROOT)
    real, _ = load_real()
    series = {"real": real[:, :, :h]}
    for m, t in MODELS.items():
        a = load_model(t)
        if a is not None:
            series[m] = a[:N_WIN, :, :h]

    os.makedirs(CACHE, exist_ok=True)
    for k, A in series.items():
        p = _cache_path(h, k)
        if os.path.exists(p):
            continue                     # a previous worker already got this one
        parts = []
        for i in range(0, A.shape[0], 100):
            X = np.ascontiguousarray(A[i:i + 100], dtype=np.float32)
            parts.append(TG.Compute_PNL(TG.Tensor(X)).cpu().detach().numpy())
        arr = np.concatenate(parts, axis=0)
        tmp = p + ".partial"
        with open(tmp, "wb") as fh:      # write then rename, never a half file
            np.save(fh, arr)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
        print(f"    h={h} {k}: {arr.shape} cached", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def pnl_cached(h, tries=14):
    """Load one horizon's P&L, spawning workers until every series is cached.

    A worker usually finishes several series before the heap catches up with it,
    so the retry count only has to exceed the number of series. The long horizons
    crash soonest and need the most attempts.
    """
    keys = ["real"] + list(MODELS)
    for _ in range(tries):
        missing = [k for k in keys if not os.path.exists(_cache_path(h, k))]
        if not missing:
            break
        print(f"  h={h}: {len(missing)} series to compute")
        subprocess.run([sys.executable, os.path.abspath(__file__),
                        "--pnl-worker", str(h)], check=False, cwd=PROJECT)
    out = {k: np.load(_cache_path(h, k)) for k in keys
           if os.path.exists(_cache_path(h, k))}
    if "real" not in out:
        raise RuntimeError(f"no real P&L at h={h}; delete {CACHE} and retry")
    return out


def re_at(pr, pf, alpha, row):
    import tailgan_eval as TE
    rv = TE.empirical_var_es(pr, alpha)[row]
    fv = TE.empirical_var_es(pf, alpha)[row]
    return float(np.nanmean(np.abs(rv - fv) / (np.abs(rv) + EPS)))


def fig4():
    sys.path.insert(0, os.path.join(ROOT, "modules", "TailGAN"))
    pnl = {h: pnl_cached(h) for h in HS}
    order = [m for m in order_models() if m in pnl[HS[0]]]

    fig, axes = plt.subplots(2, 1, figsize=(COL, 2.42), sharex=True)
    fig.subplots_adjust(left=0.165, right=0.985, top=0.815, bottom=0.155,
                        hspace=0.14)
    for ax, row, lab in zip(axes, (0, 1),
                            [r"RE$_{\mathrm{VaR}}$", r"RE$_{\mathrm{ES}}$"]):
        for m in order:
            y = [re_at(pnl[h]["real"], pnl[h][m], ALPHA, row) for h in HS]
            ax.plot(HS, y, color=C[m], lw=1.2 if m == "QuantDiffusion" else 0.8,
                    marker="o", ms=1.9, mew=0, label=m, zorder=3)
        ax.set_ylabel(f"{lab} ({ALPHA})", fontsize=6.2, labelpad=1.5)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(labelsize=5.4, pad=1.2)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
    axes[1].set_xlabel(r"horizon $h$ (trading days)", fontsize=6.2, labelpad=1.2)
    # one boxed row above the panels: Tail-GAN leaves the top of the first panel
    # at the short horizons, so anything placed inside it collides
    relabel(axes[0], ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02),
            frameon=True, fancybox=False, edgecolor="#8A8A8A",
            facecolor="white", framealpha=1.0, fontsize=4.9,
            handlelength=0.9, handletextpad=0.3, columnspacing=0.55,
            borderpad=0.25, borderaxespad=0.0)
    axes[1].text(0.02, 0.05, "mean over all 68 P&L columns", fontsize=5.0,
                 color="#666666", transform=axes[1].transAxes)
    save(fig, "fig4_risk_horizon.png")


# ==========================================================================
# fig 5: architecture, a temporary raster stand in for the draw.io figure
# ==========================================================================
def fig5():
    """Take the draw.io export as it is.

    The architecture figure is authored in draw.io, so the file that goes in the
    paper is the export from draw.io itself, not a redrawing. This only copies it
    into place under the figN naming, and refuses to invent a substitute if the
    export is missing.
    """
    import shutil

    src = os.path.join(ROOT, "result", "images", "drawio",
                       "cand_A_dmstcn_backbone.png")
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"no draw.io export at {src}. Open cand_A_dmstcn_backbone.drawio "
            "and export as PNG at the highest dpi, then rerun.")
    dst = os.path.join(OUT, "fig5_architecture.png")
    shutil.copyfile(src, dst)
    print(f"  copied the draw.io export -> {os.path.relpath(dst, ROOT)}")


def _fig5_unused():
    fig = plt.figure(figsize=(FULL, 1.42))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 21.5)
    ax.axis("off")

    stages = [
        ("Log returns", r"$r \in \mathbb{R}^{6\times256}$", "#F2F2EE"),
        ("RV decomposition", r"$r_t = z_t\sqrt{RV_t}$", "#E7EDF3"),
        ("Gaussianization", r"Lambert $W^{-1}$", "#F2EEE6"),
        ("Diffusion", r"DMS-TCN $\epsilon_\theta(x_t,t)$", "#E7EDF3"),
        ("Reconstruction", r"Lambert $W$, denorm", "#F2EEE6"),
        ("Scenarios", r"$\hat{r} \in \mathbb{R}^{6\times256}$", "#F2F2EE"),
    ]
    w, gap, y, hgt = 14.0, 3.2, 9.4, 8.4
    centres = []
    for i, (name, sub, fill) in enumerate(stages):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y), w, hgt,
                                    boxstyle="round,pad=0,rounding_size=1.1",
                                    fc=fill, ec="#6E6E68", lw=0.9, zorder=2))
        ax.text(x + w / 2, y + hgt * 0.63, name, ha="center", va="center",
                fontsize=7.8, zorder=3)
        ax.text(x + w / 2, y + hgt * 0.27, sub, ha="center", va="center",
                fontsize=6.9, color="#333333", zorder=3)
        centres.append(x + w / 2)
        if i:
            ax.add_patch(FancyArrowPatch((x - gap, y + hgt / 2),
                                         (x - 0.6, y + hgt / 2),
                                         arrowstyle="-|>", mutation_scale=7,
                                         lw=0.9, color="#4A4A4A", zorder=3,
                                         shrinkA=0, shrinkB=0))

    for i, txt in enumerate([r"$6\times256$", r"$12\times256$",
                             r"$12\times256$", r"$12\times256$",
                             r"$6\times256$"]):
        ax.text((centres[i] + centres[i + 1]) / 2, y + hgt + 1.4, txt,
                ha="center", va="bottom", fontsize=6.0, color="#777777")

    ax.text(50, 6.4, "three dilated blocks,  "
            r"$k = 15,\,7,\,5$,   $d = 1,\,4,\,16$,   128 channels,  "
            "receptive field 103", ha="center", va="top", fontsize=6.9,
            color="#333333")
    ax.text(50, 3.0, r"cosine schedule,  $\tau = 1.5$,  $T = 75$,  "
            r"Student-$t$ prior,  df $= 5$", ha="center", va="top",
            fontsize=6.9, color="#333333")
    save(fig, "fig5_architecture.png")


# ==========================================================================
FIGS = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5}

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--pnl-worker":
        pnl_worker(int(sys.argv[2]))
    want = [int(a) for a in sys.argv[1:] if a.isdigit()] or sorted(FIGS)
    for k in want:
        print(f"fig {k}")
        FIGS[k]()
