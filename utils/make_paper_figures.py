"""The figures that go in the paper.

  fig1  stylized facts:  ACF(|r|), ACF(r^2), cross asset comovement of x_t
  fig2  daily log return distributions for JPM, the figure already in main.tex,
        restyled
  fig4  risk error at every horizon, the Tail-GAN metric by h
  fig6  fig4 resolved per trading strategy block
  fig7  ablation generation fidelity, a heat map of the ratio to the full model
  fig8  regime shift: HSI 100 day cumulative returns, training and QuantDiffusion
        against the test windows

The framework figure (fig_framework.png) is exported from the slide deck and is
not drawn here.

Everything is sized for acmart sigconf: 7.0 in across both columns, 3.33 in for
one, Times New Roman to match the body text, 400 dpi.

The figures are written straight into result/images.

Run from the repository root:
      python utils/make_paper_figures.py          all six
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
from matplotlib.ticker import MaxNLocator

# This module lives in utils/ while the plotting helpers it draws on live beside
# the older figure candidates, so both paths are resolved from this file rather
# than from the working directory: the notebook, a shell in the repository root
# and the pnl worker subprocess all import it the same way.
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(_HERE)
HELPERS = os.path.join(PROJECT, "result", "images", "figures")

# When this file is run as a script (the pnl worker subprocess does exactly that:
# python utils/make_paper_figures.py --pnl-worker H), sys.path[0] is utils/, not the
# repository root, so "import utils.*" below fails with "utils is not a package".
# Put the project root on the path so the subprocess imports the same way a notebook does.
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

from utils.make_figures import (
    ASSETS,
    SEQ_LEN,
    C,
    DPI,
    EPS,
    MODELS,
    ROOT,  # noqa: E402
    RV_WINDOW,
    ST,
    acf_batch,
    flat_windows,
    load_model,
    load_real,
    read_pm_table,
    tag,
)
from utils.make_square_figures import COL_ABL  # noqa: E402

# the figures land directly under result/images
OUT = os.path.join(PROJECT, "result", "images")
# Inside the helpers directory, which .gitignore covers, because these 6.7 MB of
# intermediate arrays are a side effect of the subprocess workaround below and
# have no business being tracked next to the figures they produce.
CACHE = os.path.join(HELPERS, "pnl_cache")
os.makedirs(OUT, exist_ok=True)
FULL, COL = 7.0, 3.33  # acmart sigconf: both columns, one column
HS = [20, 40, 64, 96, 128, 160, 200, 256]
ALPHA = 0.05
N_WIN = 400  # scenarios per model where a subset suffices

# Times New Roman everywhere, with a serif math face to match it
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.6,
        "axes.labelsize": 7.6,
        "axes.titlesize": 8.2,
        "legend.fontsize": 6.8,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
    }
)


def save(fig, name, outdir=OUT, transparent=True):
    """Write at DPI with a transparent background.

    The page is white anyway, so a transparent canvas costs nothing and lets the
    figure sit on a coloured background in a slide without a white box around it.

    ``transparent=False`` bakes the white in instead, for a viewer that composites
    the alpha channel against a dark background and renders the axis labels
    unreadable.
    """
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, name)
    if not transparent:
        fig.patch.set_facecolor("white")
    fig.savefig(p, dpi=DPI, transparent=transparent)
    plt.close(fig)
    print("  wrote", os.path.relpath(p, ROOT))


# the one order every figure uses, matching the columns of the result tables
ORDER = ["QuantGAN", "TailGAN", "Diffusion-TS", "QuantDiffusion"]
# the keys are the folder names; the paper spells the model Tail-GAN
DISPLAY = {"TailGAN": "Tail-GAN"}
SHORT = {
    "QuantGAN": "QuantGAN",
    "TailGAN": "Tail-GAN",
    "Diffusion-TS": "Diffusion-TS",
    "QuantDiffusion": "ours",
}


def relabel(ax, reverse=False, **kw):
    """Draw the legend with the paper's spelling of each model."""
    h, l = ax.get_legend_handles_labels()
    l = [DISPLAY.get(x, x) for x in l]
    if reverse:
        h, l = h[::-1], l[::-1]
    ax.legend(h, l, **kw)


def order_models():
    return [m for m in ORDER if m in MODELS]


# ==========================================================================
# fig 1: stylized facts, three panels in a row
# ==========================================================================
def log_rv(A, w=RV_WINDOW):
    """x_t = log RV_t per asset, the second half of the diffusion representation."""
    c = np.cumsum(A**2, axis=-1)
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


def fig1(panel_c="cdf_tail", name="fig1_stylized_facts.png", outdir=OUT):
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
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(COL, 1.62),
        sharey=False,
        gridspec_kw={"width_ratios": [1, 1, 0.86]},
    )
    fig.subplots_adjust(left=0.125, right=0.99, top=0.755, bottom=0.235, wspace=0.30)

    panels = [
        (
            r"ACF$(|r_\tau|)$",
            lambda A: acf_batch(np.abs(flat_windows(A, N_WIN)), nlags).mean(0),
        ),
        (
            r"ACF$(r_\tau^2)$",
            lambda A: acf_batch(flat_windows(A, N_WIN) ** 2, nlags).mean(0),
        ),
    ]
    for ax, (title, fn), letter in zip(axes, panels, "ab"):
        for k, A in series.items():
            ax.plot(
                np.arange(1, nlags + 1),
                fn(A),
                color=C[k],
                lw=1.1 if k == "Real" else 0.75,
                ls="-" if k == "Real" else "--",
                label=k if letter == "a" else None,
                zorder=3,
            )
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
        m[k] = (
            comovement_windows(A)
            if k == "Real"
            else np.concatenate(
                [
                    comovement_windows(load_model(MODELS[k], r))
                    for r in range(5)
                    if load_model(MODELS[k], r) is not None
                ]
            )
        )

    if panel_c == "risk_cdf":
        # RISK view: equal-weight portfolio cumulative log return per scenario (pooled
        # over the 5 generated sets), as a CDF on a log vertical axis so the lower tail,
        # the crisis losses that VaR and ES measure, opens up and each model's fit to the
        # real loss tail is visible.
        rc = {}
        for k in series:
            if k == "Real":
                rc[k] = series[k].mean(axis=1).sum(axis=1)
            else:
                sets = [load_model(MODELS[k], r) for r in range(5)]
                rc[k] = np.concatenate(
                    [s.mean(axis=1).sum(axis=1) for s in sets if s is not None]
                )
        grid = np.linspace(0.002, 0.998, 400)
        for k in series:
            ax.plot(
                np.quantile(rc[k], grid),
                grid,
                color=C[k],
                lw=1.1 if k == "Real" else 0.75,
                ls="-" if k == "Real" else "--",
                zorder=3,
            )
        ax.set_yscale("log")
        ax.set_ylim(2e-3, 1)
        ax.set_yticks([1e-2, 1e-1, 1])
        ax.set_yticklabels(["1%", "10%", "100%"])
        ax.set_xlabel("portfolio return", fontsize=5.8, labelpad=0.8)
        ax.set_ylabel("fraction below", fontsize=5.8, labelpad=1.2)
    elif panel_c in ("cdf", "survival", "tail", "cdf_tail"):
        # drawn through a few hundred quantiles rather than all 10,000 points,
        # which is the same curve at a fraction of the file size
        top = 1.0 if panel_c == "cdf" else 0.995 if panel_c == "cdf_tail" else 0.99
        grid = np.linspace(0.0, top, 400)
        for k in series:
            q = np.quantile(m[k], grid)
            if panel_c == "cdf":
                xy = (q, grid)
            elif panel_c == "survival":
                xy = (q, 1.0 - grid)
            elif panel_c == "cdf_tail":
                # CDF with the upper tail stretched: y = -log10(1 - F), so the
                # crisis end (F -> 1, high comovement) opens up and each model's
                # fit there is visible, while the bulk compresses toward the axis
                xy = (q, -np.log10(1.0 - grid + 1e-12))
            else:
                xy = (1.0 - grid, q)
            ax.plot(
                *xy,
                color=C[k],
                lw=1.1 if k == "Real" else 0.75,
                ls="-" if k == "Real" else "--",
                zorder=3,
            )
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
        elif panel_c == "cdf_tail":
            ax.set_xlim(-0.05, 0.85)
            ax.set_xticks([0.0, 0.4, 0.8])
            ax.set_yticks([-np.log10(1 - v) for v in (0.5, 0.9, 0.99)])
            ax.set_yticklabels(["50%", "90%", "99%"])
            ax.set_ylim(0, 2.35)
            ax.set_xlabel("comovement", fontsize=5.8, labelpad=0.8)
            ax.set_ylabel("CDF", fontsize=5.8, labelpad=1.2)
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
            ax.text(
                0.97,
                0.03,
                r"$x_t = \log RV_t$",
                transform=ax.transAxes,
                fontsize=5.0,
                va="bottom",
                ha="right",
                color="#444444",
            )
    else:
        ms = [k for k in series if k != "Real"]
        ax.bar(
            range(len(ms)),
            [m[k].mean() for k in ms],
            0.66,
            color=[C[k] for k in ms],
            lw=0,
            zorder=3,
        )
        ax.plot(
            range(len(ms)),
            [np.quantile(m[k], 0.95) for k in ms],
            "D",
            ms=2.2,
            mfc="white",
            mew=0.7,
            mec=C["Real"],
            ls="none",
            zorder=5,
        )
        rm, rq = m["Real"].mean(), np.quantile(m["Real"], 0.95)
        ax.axhline(rm, lw=0.9, color=C["Real"], zorder=4)
        ax.axhline(rq, lw=0.9, ls=(0, (2, 1.4)), color=C["Real"], zorder=4)
        for y, t in [(rm, "mean"), (rq, "q95")]:
            ax.text(
                len(ms) - 0.45,
                y,
                t,
                fontsize=4.4,
                va="bottom",
                ha="right",
                color=C["Real"],
            )
        ax.set_xticks([])  # the bar colours carry the identity
        ax.set_xlim(-0.7, len(ms) - 0.3)
        ax.set_ylim(0, 0.88)
        ax.set_yticks([0.0, 0.4, 0.8])
        ax.set_xlabel("comovement", fontsize=5.8, labelpad=0.8)
    ax.set_title(
        "Risk CDF" if panel_c == "risk_cdf" else r"Corr$(x_i, x_j)$",
        pad=2.0,
        fontsize=6.4,
    )
    ax.tick_params(labelsize=5.4, pad=1.2)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    tag(ax, "c", dx=-0.30, dy=0.99)

    # one boxed row above the panels, as in fig4. It is attached to the figure
    # rather than to axes[0], which is only a third of the width here, so that it
    # centres over all three panels.
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(
        h,
        [DISPLAY.get(x, x) for x in l],
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=True,
        fancybox=False,
        edgecolor="#8A8A8A",
        facecolor="white",
        framealpha=1.0,
        fontsize=4.8,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.6,
        borderpad=0.28,
        borderaxespad=0.0,
    )
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


def fig2(asset=2, h=1):
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
    fig.subplots_adjust(
        left=0.115, right=0.985, top=0.915, bottom=0.135, wspace=0.30, hspace=0.40
    )
    flat = axes.ravel()
    for k, (ax, m) in enumerate(zip(flat, order)):
        syn = cum_h(load_model(MODELS[m]), asset, h)
        ax.hist(
            hist,
            bins=bins,
            density=True,
            color=HIST_FC,
            alpha=0.62,
            edgecolor=HIST_EC,
            lw=0.2,
            label="Historical",
            zorder=2,
        )
        ax.hist(
            syn,
            bins=bins,
            density=True,
            color=SYN_FC,
            alpha=0.62,
            edgecolor=SYN_EC,
            lw=0.2,
            label="Synthetic",
            zorder=3,
        )
        ax.set_title(DISPLAY.get(m, m), pad=2.5, fontsize=7.0, fontweight="bold")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(labelsize=5.8)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        if k % 2 == 0:
            ax.set_ylabel("Density", fontsize=6.6, labelpad=1.5)
        if k >= 2:
            ax.set_xlabel(
                "Daily log return" if h == 1
                else f"Cumulative log return ({h} Day)",
                fontsize=6.6,
                labelpad=1.5,
            )
    flat[0].legend(
        loc="upper left",
        fontsize=4.8,
        handlelength=0.8,
        labelspacing=0.18,
        borderpad=0.18,
        handletextpad=0.4,
    )
    save(fig, "fig2_distribution.png")


# ==========================================================================
# fig 4: risk error at every horizon
# ==========================================================================
# fig4 must equal Table portfolio_risk at its terminal horizon, so it is computed
# exactly as notebook 2's vares_grid_oos: the same five OOS baskets
# (data/our_basket) and five seeds, each basket's transition matrix read directly
# and fed to compute_pnl_direct (bit-for-bit identical to TG.Compute_PNL, no live
# slot swap). h=256 (=SEQ_LEN) is the untruncated terminal that matches the table.
_FIG_BASKET_DIR = os.path.join(PROJECT, "data", "our_basket")
N_BASKETS = 10
N_SEEDS = 5
# the 68 Compute_PNL columns in order: BuyHold[0:6], static portfolios[6:56],
# mean reversion[56:62], trend following[62:68]. "all" is what fig4 reports;
# the three named strategies are the columns of fig6.
_STRATS = {
    "all": slice(None),
    "Port": slice(6, 56),
    "MR": slice(56, 62),
    "TF": slice(62, 68),
}


def _cache_path(h, b):
    return os.path.join(CACHE, f"h{h}_b{b}_re.npy")


def pnl_worker(h):
    """VaR/ES relative error at one horizon, computed exactly as Table
    portfolio_risk and notebook 2's vares_grid_oos. For each of the five OOS
    baskets (data/our_basket) the basket's transition matrix is read directly and
    fed to compute_pnl_direct (bit-for-bit identical to TG.Compute_PNL, no live
    slot swap); real and every seed batch are truncated to [:h] and scored with
    the same tailgan_eval.relerror the table uses; the per basket mean over the
    five seeds is cached as h{h}_b{b}_re.npy.

    Compute_PNL/torch can die inside the interpreter at teardown, so each basket
    is written the moment it is done and a rerun resumes from what survived; the
    worker leaves through os._exit to skip teardown.
    """
    sys.path.insert(0, os.path.join(ROOT, "modules", "TailGAN"))
    import tailgan_eval as TE
    from utils.tailgan_direct import compute_pnl_direct

    TG = TE.configure_from_project(ROOT)
    real, _ = load_real()
    seeds = {}
    for m, t in MODELS.items():
        arrs = [load_model(t, s) for s in range(N_SEEDS)]
        arrs = [a for a in arrs if a is not None]
        if arrs:
            seeds[m] = arrs

    def pnl(A, tm):  # whole batch, horizon-truncated, tm read directly
        return (
            compute_pnl_direct(
                TG, TG.Tensor(np.asarray(A[:, :, :h], dtype=np.float32)), tm
            )
            .cpu()
            .detach()
            .numpy()
        )

    os.makedirs(CACHE, exist_ok=True)
    for b in range(N_BASKETS):
        p = _cache_path(h, b)
        if os.path.exists(p):
            continue
        tm = np.load(os.path.join(_FIG_BASKET_DIR, f"basket{b}", "TransMat_OOS.npy"))
        rp = pnl(real, tm)
        assert rp.shape[1] == 68, f"expected 68 P&L columns, got {rp.shape[1]}"
        out = {st: {} for st in _STRATS}
        for m, arrs in seeds.items():
            acc = {st: [[], []] for st in _STRATS}
            for a in arrs:
                fp = pnl(a, tm)
                for st, cols in _STRATS.items():
                    d = TE.relerror(rp[:, cols], fp[:, cols], (ALPHA,))
                    acc[st][0].append(d[f"RE_VaR({ALPHA})"])
                    acc[st][1].append(d[f"RE_ES({ALPHA})"])
            for st in _STRATS:
                out[st][m] = [
                    float(np.mean(acc[st][0])),
                    float(np.mean(acc[st][1])),
                ]
        tmp = p + ".partial"
        with open(tmp, "wb") as fh:  # write then rename, never a half file
            np.save(fh, np.array(out, dtype=object), allow_pickle=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
        print(f"    h={h} basket{b}: {list(out)} cached", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def pnl_cached(h, tries=20):
    """Relative error per model at horizon h, averaged over the five baskets.

    Spawns the worker until every basket is cached, then averages the per basket
    means so the terminal horizon equals Table portfolio_risk.
    """
    for _ in range(tries):
        missing = [b for b in range(N_BASKETS) if not os.path.exists(_cache_path(h, b))]
        if not missing:
            break
        print(f"  h={h}: {len(missing)} baskets to compute")
        subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--pnl-worker", str(h)],
            check=False,
            cwd=PROJECT,
        )
    per_basket = [
        np.load(_cache_path(h, b), allow_pickle=True).item()
        for b in range(N_BASKETS)
        if os.path.exists(_cache_path(h, b))
    ]
    if not per_basket:
        raise RuntimeError(f"no P&L at h={h}; delete {CACHE} and retry")
    out = {}
    for st in per_basket[0]:
        out[st] = {}
        for m in per_basket[0][st]:
            out[st][m] = (
                float(np.mean([d[st][m][0] for d in per_basket])),
                float(np.mean([d[st][m][1] for d in per_basket])),
            )
    return out


def fig4():
    pnl = {h: pnl_cached(h) for h in HS}
    order = [m for m in order_models() if m in pnl[HS[0]]["all"]]

    # same canvas and type sizes as fig6, so the two risk figures read as a
    # pair rather than as two different chart styles
    fig, axes = plt.subplots(2, 1, figsize=(COL, 1.55), sharex=True)
    fig.subplots_adjust(left=0.165, right=0.985, top=0.795, bottom=0.235,
                        hspace=0.18)
    for ax, row, lab in zip(
        axes, (0, 1), [r"RE$_{\mathrm{VaR}}$", r"RE$_{\mathrm{ES}}$"]
    ):
        for m in order:
            y = [pnl[h]["all"][m][row] for h in HS]
            ax.plot(
                HS,
                y,
                color=C[m],
                lw=1.2 if m == "QuantDiffusion" else 0.8,
                marker="o",
                ms=1.9,
                mew=0,
                label=m,
                zorder=3,
            )
        # the alpha lives in the caption, as in fig6
        ax.set_ylabel(lab, fontsize=5.6, labelpad=1.4)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(labelsize=5.0, pad=1.0)
        ax.grid(axis="y", lw=0.4)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
    axes[1].set_xlabel(r"horizon $h$ (trading days)", fontsize=5.2, labelpad=1.2)
    # one boxed row above the panels: Tail-GAN leaves the top of the first panel
    # at the short horizons, so anything placed inside it collides
    relabel(
        axes[0],
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.06),
        frameon=True,
        fancybox=False,
        edgecolor="#8A8A8A",
        facecolor="white",
        framealpha=1.0,
        fontsize=4.8,
        handlelength=0.8,
        handletextpad=0.3,
        columnspacing=0.55,
        borderpad=0.25,
        borderaxespad=0.0,
    )
    save(fig, "fig4_risk_horizon.png")


# ==========================================================================
# fig 6: fig4 resolved per trading strategy
# ==========================================================================
def fig6():
    """Risk error by trading strategy, averaged over the horizon grid.

    The four groups are the blocks of Compute_PNL's sixty eight columns: buy and
    hold on the six assets, the fifty static portfolios, mean reversion and trend
    following. Buy and hold is not cached on its own, but relerror is a plain mean
    over columns, so the block recombines exactly:

        BH = (68 * all - 50 * Port - 6 * MR - 6 * TF) / 6

    The bar is the mean over the eight horizons and the whisker spans h = 20 to
    h = 256, because the error grows with the horizon and a bare average would sit
    below the terminal value the risk table reports. Group sizes are printed beside
    each name since the fifty static portfolios dominate the pooled column.
    """
    pnl = {h: pnl_cached(h) for h in HS}
    order = [m for m in order_models() if m in pnl[HS[0]]["all"]]
    NCOL = {"BH": 6, "Port": 50, "MR": 6, "TF": 6}
    groups = [("BH", "Buy and hold"), ("Port", "Static portfolios"),
              ("MR", "Mean reversion"), ("TF", "Trend following")]

    def val(h, key, m, row):
        if key != "BH":
            return pnl[h][key][m][row]
        a, p_, mr, tf = (pnl[h][k][m][row] for k in ("all", "Port", "MR", "TF"))
        return (68.0 * a - 50.0 * p_ - 6.0 * mr - 6.0 * tf) / 6.0

    # half the previous height, so the two panels each get about half an inch and
    # the ticks and the group labels have to shrink with them
    fig, axes = plt.subplots(2, 1, figsize=(COL, 1.55), sharex=True)
    fig.subplots_adjust(left=0.165, right=0.985, top=0.795, bottom=0.235,
                        hspace=0.18)
    x = np.arange(len(groups))
    w = 0.20
    for row, (ax, lab) in enumerate(zip(axes, [r"RE$_{\mathrm{VaR}}$",
                                              r"RE$_{\mathrm{ES}}$"])):
        for k, m in enumerate(order):
            mean = [np.mean([val(h, g, m, row) for h in HS]) for g, _ in groups]
            lo = [min(val(h, g, m, row) for h in HS) for g, _ in groups]
            hi = [max(val(h, g, m, row) for h in HS) for g, _ in groups]
            pos = x + (k - (len(order) - 1) / 2) * w
            ax.bar(pos, mean, w * 0.88, color=C[m], lw=0, zorder=3,
                   label=m if row == 0 else None)
            ax.vlines(pos, lo, hi, color="#3A3A3A", lw=0.55, zorder=4)
            ax.hlines(hi, pos - w * 0.22, pos + w * 0.22, color="#3A3A3A",
                      lw=0.55, zorder=4)
            ax.hlines(lo, pos - w * 0.22, pos + w * 0.22, color="#3A3A3A",
                      lw=0.55, zorder=4)
        # the alpha is in the caption, so the axis carries only the measure name;
        # at half the height a longer label would collide with the other panel
        ax.set_ylabel(lab, fontsize=5.6, labelpad=1.4)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(labelsize=5.0, pad=1.0)
        ax.grid(axis="y", lw=0.4)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [f"{n}\n$n={NCOL[g]}$" for g, n in groups], fontsize=5.2)
    axes[1].tick_params(axis="x", pad=1.6)
    relabel(
        axes[0],
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.06),
        frameon=True,
        fancybox=False,
        edgecolor="#8A8A8A",
        facecolor="white",
        framealpha=1.0,
        fontsize=5.4,
        handlelength=0.9,
        handletextpad=0.3,
        columnspacing=0.7,
        borderpad=0.25,
        borderaxespad=0.0,
    )
    save(fig, "fig6_risk_horizon_strategies.png")


# ==========================================================================
# fig 7: the generation fidelity of the ablation variants, relative to Full
# ==========================================================================
# Six of the nineteen fidelity metrics, one per property the design targets:
# cross asset dependence, the one day marginal, its asymmetry, its tail, and
# volatility clustering in two forms. ACF(|r|) is used rather than ACF(r)
# because ACF(r) is close to zero in the data, so that axis rewards a model for
# generating no structure at all and inverts its own meaning.
FIDELITY6 = [
    ("CCF(τ = 0)_MAE", "CCF\n$\\tau$=0"),
    ("KS(1)", "KS\n(1)"),
    ("Skew(1)", "Skew\n(1)"),
    ("Kurt(1)", "Kurt\n(1)"),
    ("ACF(x^2)", "ACF\n$r^{2}$"),
    ("ACF(|x|)", "ACF\n$|r|$"),
]
# ordered so the two variants that only lose on risk sit above the two that lose
# on everything, which puts the blue block and the red block side by side
FIG7_ROWS = ["Full", "w/o Lambert W", "w/o L1 loss", "w/o cosine",
             "w/o DMS-TCN", "w/o RV"]


def fig7(risk_col="var", height=1.40):
    """Ablation fidelity as a heat map of the ratio to the full model.

    It is a heat map rather than a radar because a radar has to normalize each
    axis to a radius and every available normalization is either set dependent
    (per axis min-max moves a polygon when a *different* variant is added or
    removed) or references a quantity nobody asked about (the axis mean). The
    ratio to Full is what the ablation is about, so it is encoded directly:
    diverging around an exact neutral at Full, blue better and red worse. The
    colour bar is inverted so that better is at the top.

    ``risk_col`` rules a risk column off from the fidelity block: "var" averages
    the three VaR levels, "all" averages all six VaR and ES entries, None drops
    it. Dropping it costs the contrast that motivates the figure, since the
    fidelity block reads mostly blue while the risk column is red throughout.
    """
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    label = {"var": "VaR RE\nmean", "all": "risk\nmean RE"}
    fid, _ = read_pm_table(os.path.join(ST, "ablation__gen_quality.csv"))
    risk, _ = read_pm_table(os.path.join(ST, "ablation__tailgan_vares.csv"))
    ren = {COL_ABL[k]: k for k in COL_ABL}
    fid, risk = fid.rename(columns=ren), risk.rename(columns=ren)
    if risk_col == "var":
        risk = risk.loc[[i for i in risk.index if i.startswith("RE_VaR")]]

    cols = list(FIDELITY6) + ([("__risk__", label[risk_col])] if risk_col else [])
    Z = np.empty((len(FIG7_ROWS), len(cols)))
    for j, (k, _) in enumerate(cols):
        base = risk.mean()["Full"] if k == "__risk__" else fid.loc[k, "Full"]
        for i, m in enumerate(FIG7_ROWS):
            v = risk.mean()[m] if k == "__risk__" else fid.loc[k, m]
            Z[i, j] = np.log2(v / base)

    cmap = LinearSegmentedColormap.from_list(
        "bwr_paper", ["#234EB6", "#7FA0D6", "#EDEDEA", "#DFA08C", "#B4341A"])
    norm = TwoSlopeNorm(vmin=-1.2, vcenter=0.0, vmax=1.2)

    fig, (ax, cax) = plt.subplots(
        1, 2, figsize=(COL, height),
        gridspec_kw={"width_ratios": [1, 0.03], "wspace": 0.03})
    # the two line column headers need a fixed band in inches, so the top margin
    # is derived from the height rather than fixed as a fraction
    fig.subplots_adjust(left=0.21, right=0.89, top=1.0 - 0.34 / height, bottom=0.04)
    ax.imshow(Z, cmap=cmap, norm=norm, aspect="auto")
    # the surface gap the mark spec asks for between adjacent filled cells
    for x in np.arange(-0.5, len(cols), 1):
        ax.axvline(x, color="white", lw=1.4)
    for y in np.arange(-0.5, len(FIG7_ROWS), 1):
        ax.axhline(y, color="white", lw=1.4)
    if risk_col:
        ax.axvline(len(cols) - 1.5, color="#444444", lw=1.1)

    for i in range(len(FIG7_ROWS)):
        for j in range(len(cols)):
            r = 2.0 ** Z[i, j]
            txt = "—" if FIG7_ROWS[i] == "Full" else (
                f"{r:.2f}" if r < 10 else f"{r:.1f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.0,
                    color="#1A1A19" if abs(Z[i, j]) < 0.75 else "white")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([h for _, h in cols], fontsize=5.4)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(FIG7_ROWS)))
    ax.set_yticklabels(FIG7_ROWS, fontsize=5.8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # the bar carries the reading, so there is no "lower is better" caption to
    # parse: blue on top is better than Full, red at the bottom is worse. That
    # means inverting the bar, since the value it encodes runs the other way.
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_ticks([-1.0, 0.0, 1.0])
    cb.set_ticklabels(["better", "= Full", "worse"])
    cb.ax.invert_yaxis()
    cb.ax.tick_params(labelsize=5.0, length=0)
    cb.outline.set_visible(False)
    save(fig, "fig7c_heatmap.png")


# ==========================================================================
# fig 8: regime shift between the training and test partitions
# ==========================================================================
def fig8(asset=3, h=100, name="fig8_regime_tails.png", outdir=OUT):
    """Regime shift between the partitions, drawn like fig2: overlapping filled histograms
    of the h-day cumulative log return of one asset (HSI over 100 days by default, the
    asset where the shift between the partitions is clearest).

    (a) training windows against test windows, (b) QuantDiffusion against test windows.
    The camera-ready answer to Reviewer 2: panel (a) is the shift between the two
    partitions that every out-of-sample risk error contains; panel (b) places our model
    against the same test distribution. Only our model is drawn.
    """
    import pandas as pd

    df = pd.read_csv(os.path.join(PROJECT, "data", "indices.csv"), index_col=0)
    r = np.log1p(df.apply(pd.to_numeric).astype(float).pct_change()).dropna().values
    n = int(0.7 * len(r))

    def windows(a):  # (T, d) -> (N, d, L)
        return np.lib.stride_tricks.sliding_window_view(a, SEQ_LEN, axis=0)

    train_w, test_w = windows(r[:n]), windows(r[n:])
    ours = np.concatenate([np.load(os.path.join(ST, "synthetic_data", "baseline", f"{s}.npy"))
                           for s in range(5)])
    test = cum_h(test_w, asset, h, n_win=len(test_w))
    other = {"Training": cum_h(train_w, asset, h, n_win=len(train_w)),
             "QuantDiffusion": cum_h(ours, asset, h, n_win=len(ours))}
    bins = np.linspace(*np.percentile(np.r_[test, other["Training"]], [0.05, 99.95]), 61)

    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.55), sharex=True)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.86, bottom=0.25, wspace=0.30)
    for ax, (label, x) in zip(axes, other.items()):
        ax.hist(test, bins=bins, density=True, color=HIST_FC, alpha=0.62, edgecolor=HIST_EC,
                lw=0.2, label="Test", zorder=2)
        ax.hist(x, bins=bins, density=True, color=SYN_FC, alpha=0.62, edgecolor=SYN_EC,
                lw=0.2, label=label, zorder=3)
        ax.set_title(f"{label} vs. Test", pad=2.5, fontsize=7.0, fontweight="bold")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(labelsize=5.8)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.set_xlabel(f"Cumulative log return ({h} Day)", fontsize=6.6, labelpad=1.5)
        ax.legend(loc="upper left", fontsize=4.8, handlelength=0.8, labelspacing=0.18,
                  borderpad=0.18, handletextpad=0.4)
    axes[0].set_ylabel("Density", fontsize=6.6, labelpad=1.5)
    for k, v in {"Test": test, **other}.items():
        print(f"  {k:15s} 5% quantile {np.quantile(v, 0.05):+.4f}  std {v.std():.4f}")
    save(fig, name, outdir=outdir)


# ==========================================================================
FIGS = {1: fig1, 2: fig2, 4: fig4, 6: fig6, 7: fig7, 8: fig8}

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--pnl-worker":
        pnl_worker(int(sys.argv[2]))
    want = [int(a) for a in sys.argv[1:] if a.isdigit()] or sorted(FIGS)
    for k in want:
        print(f"fig {k}")
        FIGS[k]()
