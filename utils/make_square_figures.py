"""Square PNG figures for the QuantDiffusion paper.

Six candidates in a 1:1 aspect, for slots where a wide figure does not fit.
They reuse the loaders, palette and rcParams of make_figures.py, so a subset
kept alongside the wide figures stays visually consistent.

  sq_1  radar over six statistical fidelity axes, four models
  sq_2  radar over the six portfolio risk errors
  sq_3  radar over the same fidelity axes for the ablation variants
  sq_4  quantile to quantile plot of one day returns against the real data
  sq_5  tail fidelity against clustering fidelity, one point per model
  sq_6  cross asset correlation, real and three baselines in a two by two grid

Metric values come from result/statistics/*.csv, that is from the same tables the
paper reports, so no number here can drift from Table 1 or Table 2. Panels that
need the samples themselves (sq_4, sq_6) read the generated .npy files.

Run:  python make_square_figures.py          all six
      python make_square_figures.py 1 5      only these
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.make_figures import (
    ASSETS,
    C,
    DPI,
    HERE,
    MODELS,
    ST,
    load_model,
    load_real,
    read_pm_table,
    tag,
)

BENCH = os.path.join(ST, "benchmark__gen_quality.csv")
BENCH_RISK = os.path.join(ST, "benchmark__tailgan_vares.csv")
ABL = os.path.join(ST, "ablation__gen_quality.csv")

COL_BENCH = {
    "QuantDiffusion": "QuantDiffusion (ours, C2C/w10)",
    "QuantGAN": "QuantGAN",
    "TailGAN": "TailGAN",
    "Diffusion-TS": "Diffusion-TS",
}
COL_ABL = {
    "Full": "Baseline",
    "w/o RV": "w/o RV Decomp.",
    "w/o Lambert W": "w/o LambertW",
    "w/o DMS-TCN": "w/o DMS-TCN",
}

# six axes spanning dependence, marginal shape, tails, clustering and leverage
FIDELITY = [
    ("CCF(τ = 0)_MAE", r"CCF($\tau$=0)"),
    ("CCF(τ > 0)_MAE", r"CCF($\tau$>0)"),
    ("KS(1)", "KS(1)"),
    ("Kurt(1)", "Kurt(1)"),
    ("ACF(|x|)", r"ACF($|x|$)"),
    ("Leverage Effect", "leverage"),
]
RISK = [
    ("RE_VaR(0.05)", r"VaR$_{0.05}$"),
    ("RE_VaR(0.025)", r"VaR$_{0.025}$"),
    ("RE_VaR(0.01)", r"VaR$_{0.01}$"),
    ("RE_ES(0.01)", r"ES$_{0.01}$"),
    ("RE_ES(0.025)", r"ES$_{0.025}$"),
    ("RE_ES(0.05)", r"ES$_{0.05}$"),
]


def save(fig, name):
    p = os.path.join(HERE, name)
    fig.savefig(p, dpi=DPI, facecolor="white")
    plt.close(fig)
    print("  wrote", os.path.relpath(p, os.path.join(HERE, "..", "..", "..")))


# --------------------------------------------------------------------------
def radar(ax, axes_spec, values, order, rmin=0.16, fmt="{:.4f}", note=True, gamma=1.0):
    """Radar with every axis inverted so that further out is better.

    All six metrics are errors, so each axis is min-max scaled across the models
    and then flipped: the best model reaches the outer ring, the worst sits at
    ``rmin``. The raw error is printed at each vertex, as in the reference figure,
    because the scaling is only a layout device and carries no meaning of its own.

    ``gamma`` warps that layout: the normalized distance from the best model is
    raised to ``gamma`` before flipping, so ``gamma < 1`` opens up the crowded
    best end and pulls apart models with nearly equal errors (e.g. Full and w/o
    Lambert W in the ablation panel). The printed values are unchanged.
    """
    labels = [lab for _, lab in axes_spec]
    n = len(labels)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    M = np.array([[values[m][k] for k, _ in axes_spec] for m in order])
    lo, hi = M.min(axis=0), M.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    R = 1.0 - ((M - lo) / span) ** gamma * (1.0 - rmin)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(ang)
    ax.set_xticklabels(labels, fontsize=7.2)
    ax.set_yticks([])
    ax.set_ylim(0, 1.30)
    ax.spines["polar"].set_linewidth(0.7)
    ax.spines["polar"].set_color("#444444")
    ax.xaxis.grid(True, color="#BBBBBB", lw=0.5)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)

    for i, m in enumerate(order):
        r = np.concatenate([R[i], R[i][:1]])
        a = np.concatenate([ang, ang[:1]])
        lw = 1.6 if m in ("QuantDiffusion", "Full") else 1.0
        ax.plot(a, r, lw=lw, color=C[m], label=m, zorder=3)
        ax.fill(a, r, color=C[m], alpha=0.10, lw=0, zorder=2)
        ax.scatter(ang, R[i], s=7, color=C[m], zorder=4, lw=0)

    # raw values at each vertex, best first. Two models with near identical
    # errors would stamp their labels on top of each other, so a minimum radial
    # separation is enforced by pushing later labels inward.
    gap, floor = 0.092, 0.10
    for j in range(n):
        rank = list(np.argsort(-R[:, j]))
        pos, prev = [], None
        for i in rank:
            rr = R[i, j] if prev is None else min(R[i, j], prev - gap)
            pos.append(rr)
            prev = rr
        # a cluster of similar errors would otherwise be pushed past the centre
        deficit = floor - min(pos)
        if deficit > 0:
            pos = [p + deficit for p in pos]
        for i, rr in zip(rank, pos):
            ax.text(
                ang[j],
                rr,
                fmt.format(M[i, j]),
                fontsize=4.9,
                color=C[order[i]],
                ha="center",
                va="center",
                zorder=5,
                bbox=dict(fc="white", ec="none", alpha=0.78, pad=0.5),
            )
    if note:
        ax.text(
            0.5,
            -0.13,
            "further from the centre is better; " "printed values are the raw errors",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=6.0,
            color="#666666",
        )


# ==========================================================================
def sq1_radar_fidelity():
    mean, _ = read_pm_table(BENCH)
    vals = {m: {k: mean.loc[k, c] for k, _ in FIDELITY} for m, c in COL_BENCH.items()}
    fig = plt.figure(figsize=(3.6, 3.6))
    ax = fig.add_subplot(projection="polar")
    radar(ax, FIDELITY, vals, list(COL_BENCH))
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(-0.16, 1.14),
        fontsize=6.4,
        handlelength=1.1,
        labelspacing=0.28,
    )
    ax.set_title("statistical fidelity", fontsize=8.4, pad=18)
    save(fig, "sq_1_radar_fidelity.png")


def sq2_radar_risk():
    mean, _ = read_pm_table(BENCH_RISK)
    vals = {m: {k: mean.loc[k, c] for k, _ in RISK} for m, c in COL_BENCH.items()}
    fig = plt.figure(figsize=(3.6, 3.6))
    ax = fig.add_subplot(projection="polar")
    radar(ax, RISK, vals, list(COL_BENCH), fmt="{:.3f}")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(-0.16, 1.14),
        fontsize=6.4,
        handlelength=1.1,
        labelspacing=0.28,
    )
    ax.set_title("portfolio risk, relative error", fontsize=8.4, pad=18)
    save(fig, "sq_2_radar_risk.png")


def sq3_radar_ablation():
    mean, _ = read_pm_table(ABL)
    vals = {m: {k: mean.loc[k, c] for k, _ in FIDELITY} for m, c in COL_ABL.items()}
    fig = plt.figure(figsize=(3.6, 3.6))
    ax = fig.add_subplot(projection="polar")
    radar(ax, FIDELITY, vals, list(COL_ABL))
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(-0.16, 1.14),
        fontsize=6.4,
        handlelength=1.1,
        labelspacing=0.28,
    )
    ax.set_title("ablation, statistical fidelity", fontsize=8.4, pad=18)
    save(fig, "sq_3_radar_ablation.png")


# ==========================================================================
def sq4_qq():
    """Quantile to quantile plot of pooled one day returns against the real data."""
    real, _ = load_real()

    def std(A, n=1500):
        v = np.asarray(A[:n], float).ravel()
        v = v[np.isfinite(v)]
        return (v - v.mean()) / v.std()

    p = np.concatenate(
        [
            np.linspace(0.0005, 0.02, 60),
            np.linspace(0.02, 0.98, 200),
            np.linspace(0.98, 0.9995, 60),
        ]
    )
    qr = np.quantile(std(real), p)

    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    lim = 9.0
    ax.plot(
        [-lim, lim],
        [-lim, lim],
        lw=0.7,
        ls="--",
        color="#888888",
        label="exact match",
        zorder=1,
    )
    for name, tg in MODELS.items():
        a = load_model(tg)
        if a is None:
            continue
        ax.plot(
            qr,
            np.quantile(std(a), p),
            lw=1.4 if name == "QuantDiffusion" else 1.0,
            color=C[name],
            label=name,
            zorder=3,
        )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("real quantile (standardized)")
    ax.set_ylabel("generated quantile (standardized)")
    ax.set_title("one day return quantiles", fontsize=8.4, pad=6)
    ax.grid(alpha=0.5, lw=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=6.4, handlelength=1.2, labelspacing=0.28)
    # the reading has to hold for both tails: away from the line in either
    # direction is a heavier tail, so the honest cue is the slope, not the side
    ax.text(
        0.98,
        0.03,
        "a flatter curve means lighter tails\n" "than the real data",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="#666666",
    )
    save(fig, "sq_4_qq_returns.png")


def sq5_tradeoff():
    """Tail fidelity against clustering fidelity; the corner is the good place."""
    mean, std = read_pm_table(BENCH)
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    for name, col in COL_BENCH.items():
        x, y = mean.loc["ACF(|x|)", col], mean.loc["Kurt(1)", col]
        ax.errorbar(
            x,
            y,
            xerr=std.loc["ACF(|x|)", col],
            yerr=std.loc["Kurt(1)", col],
            fmt="o",
            ms=6.5,
            color=C[name],
            lw=0.8,
            capsize=1.6,
            zorder=3,
            label=name,
        )
        ax.annotate(
            name,
            (x, y),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=6.6,
            color=C[name],
        )
    ax.set_xlim(0, 0.0060)
    ax.set_ylim(0, 6.0)
    ax.set_xlabel(r"$\mathrm{ACF}(|x|)$ error   (volatility clustering)")
    ax.set_ylabel(r"$\mathrm{Kurt}(1)$ error   (tail shape)")
    ax.set_title("tails against clustering", fontsize=8.4, pad=6)
    ax.grid(alpha=0.5, lw=0.4)
    ax.set_axisbelow(True)
    ax.annotate(
        "",
        xy=(0.0004, 0.42),
        xytext=(0.0021, 2.3),
        arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#666666"),
    )
    ax.text(0.0022, 2.5, "better on both", fontsize=6.4, color="#666666")
    save(fig, "sq_5_tradeoff.png")


def sq6_corr_grid():
    real, _ = load_real()
    names = ["DJIA", "NASDAQ", "JPM", "HSI", "Gold", "WTI"]

    def corr(A, n=1200):
        A = A[:n]
        Ac = A - A.mean(axis=2, keepdims=True)
        sd = A.std(axis=2) + 1e-12
        cov = np.einsum("nit,njt->nij", Ac, Ac) / A.shape[2]
        return (cov / (sd[:, :, None] * sd[:, None, :])).mean(axis=0)

    ccf, _ = read_pm_table(BENCH)
    panels = [("Real", corr(real), None)]
    for m in ["QuantDiffusion", "Diffusion-TS", "TailGAN"]:
        a = load_model(MODELS[m])
        if a is not None:
            panels.append((m, corr(a), ccf.loc["CCF(τ = 0)_MAE", COL_BENCH[m]]))

    fig, axes = plt.subplots(2, 2, figsize=(4.2, 4.4))
    fig.subplots_adjust(
        wspace=0.08, hspace=0.24, left=0.14, right=0.99, top=0.90, bottom=0.15
    )
    for idx, (ax, (name, M, err)) in enumerate(zip(axes.ravel(), panels)):
        im = ax.imshow(M, vmin=-0.3, vmax=1.0, cmap="RdBu_r")
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        # ticks only on the outer edges, so a title never lands on a tick label
        ax.set_xticklabels(names if idx >= 2 else [], rotation=90, fontsize=5.0)
        ax.set_yticklabels(names if idx % 2 == 0 else [], fontsize=5.0)
        ttl = name if err is None else f"{name}\nCCF($\\tau$=0) {err:.4f}"
        ax.set_title(
            ttl,
            fontsize=6.8,
            pad=3,
            color=C[name],
            fontweight="bold" if name == "Real" else "normal",
        )
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axes, fraction=0.020, pad=0.02)
    cb.ax.tick_params(labelsize=5.4, length=1.6)
    cb.outline.set_linewidth(0.4)
    save(fig, "sq_6_corr_grid.png")


def sq7_ccf_lag1():
    """Lagged cross correlation, which is where the paper's method actually wins.

    At tau = 0 the matrix is symmetric with ones on the diagonal, which is why
    sq_6 looks the way it does. At tau > 0 neither holds: rho_tau[i, j] =
    Corr(x_i,t , x_j,t+tau) is not symmetric, and the asymmetry is the lead lag
    structure. The diagonal is the lag tau autocorrelation of each asset.
    """
    names = ["DJIA", "NASDAQ", "JPM", "HSI", "Gold", "WTI"]

    def ccf(A, tau=1, n=1200):
        A = A[:n]
        X, Y = A[:, :, :-tau], A[:, :, tau:]
        Xc = X - X.mean(axis=2, keepdims=True)
        Yc = Y - Y.mean(axis=2, keepdims=True)
        sx, sy = X.std(axis=2) + 1e-12, Y.std(axis=2) + 1e-12
        cov = np.einsum("nit,njt->nij", Xc, Yc) / X.shape[2]
        return (cov / (sx[:, :, None] * sy[:, None, :])).mean(axis=0)

    real, _ = load_real()
    tab, _ = read_pm_table(BENCH)
    panels = [("Real", ccf(real), None)]
    for m in ["QuantDiffusion", "Diffusion-TS", "TailGAN"]:
        a = load_model(MODELS[m])
        if a is not None:
            panels.append((m, ccf(a), tab.loc["CCF(τ > 0)_MAE", COL_BENCH[m]]))

    # symmetric range taken from the data, so an overshoot in the HSI column is
    # visible instead of being clipped at the end of the colour scale
    vmax = float(np.ceil(max(np.abs(M).max() for _, M, _ in panels) * 20) / 20)

    fig, axes = plt.subplots(2, 2, figsize=(4.2, 4.8))
    fig.subplots_adjust(
        wspace=0.08, hspace=0.26, left=0.14, right=0.99, top=0.90, bottom=0.19
    )
    for idx, (ax, (name, M, err)) in enumerate(zip(axes.ravel(), panels)):
        im = ax.imshow(M, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels(names if idx >= 2 else [], rotation=90, fontsize=5.0)
        ax.set_yticklabels(names if idx % 2 == 0 else [], fontsize=5.0)
        ttl = name if err is None else f"{name}\nCCF($\\tau$>0) {err:.4f}"
        ax.set_title(
            ttl,
            fontsize=6.8,
            pad=3,
            color=C[name],
            fontweight="bold" if name == "Real" else "normal",
        )
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        # the boxed column is the structure the caption talks about; no leader
        # line, so nothing crosses the cells
        if name in ("Real", "QuantDiffusion", "Diffusion-TS"):
            ax.add_patch(
                plt.Rectangle(
                    (2.5, -0.5), 1, 3, fill=False, ec="#222222", lw=0.9, zorder=4
                )
            )
    cb = fig.colorbar(im, ax=axes, fraction=0.020, pad=0.02)
    cb.ax.tick_params(labelsize=5.4, length=1.6)
    cb.outline.set_linewidth(0.4)
    fig.text(
        0.5,
        0.015,
        "row: asset at $t$,   column: asset at $t+1$.   The matrix is not\n"
        "symmetric, and the asymmetry is the lead lag structure: the boxed\n"
        "column is the US indices leading the Hang Seng.",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color="#666666",
    )
    save(fig, "sq_7_ccf_lag1.png")


# --------------------------------------------------------------------------
def _decompose(A, w=10, n=800):
    """Apply the paper's own decomposition to any return array.

    RV_t is the mean of the squared returns over days t to t+w-1, the forward
    looking window the pipeline uses, so z_t = r_t / sqrt(RV_t) and x_t =
    log sqrt(RV_t). Nothing here is specific to QuantDiffusion: that is the point,
    it lets a baseline be examined on the two channels it never modelled.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    A = np.asarray(A[:n], float)
    rv = sliding_window_view(A**2, w, axis=2).mean(axis=-1)
    frv = np.sqrt(np.maximum(rv, 1e-16))
    return A[..., : frv.shape[-1]] / frv, np.log(frv)


def _acf_rows(X, nlags):
    Xc = X - X.mean(axis=-1, keepdims=True)
    den = (Xc**2).sum(axis=-1) + 1e-18
    return np.stack(
        [(Xc[..., :-k] * Xc[..., k:]).sum(axis=-1) / den for k in range(1, nlags + 1)],
        axis=-1,
    )


def _exkurt(v):
    v = np.asarray(v, float).ravel()
    v = v[np.isfinite(v)]
    v = v - v.mean()
    s = v.std()
    return float((v**4).mean() / s**4 - 3.0) if s > 0 else np.nan


def sq8_decomposition_diagnostic():
    """The decomposition used as a diagnostic, which only this paper would run.

    Every model is put through r -> (z, x) and judged on both channels. A
    generator that matches the marginal of r can still put the volatility in the
    wrong place, and that is invisible in any metric computed on r alone.
    """
    real, _ = load_real()
    series = {"Real": real}
    for name, tg in MODELS.items():
        a = load_model(tg)
        if a is not None:
            series[name] = a
    dec = {k: _decompose(v) for k, v in series.items()}

    fig, axes = plt.subplots(2, 2, figsize=(3.9, 3.9))
    fig.subplots_adjust(
        wspace=0.34, hspace=0.52, left=0.13, right=0.98, top=0.90, bottom=0.12
    )
    NL = 40

    def std(v):
        v = np.asarray(v, float).ravel()
        v = v[np.isfinite(v)]
        return (v - v.mean()) / (v.std() + 1e-18)

    # (a) comovement of the volatility channels across assets. Section 5.2 claims
    #     RV_t is what transmits cross asset risk, and this measures exactly that.
    #     The shock channel is left out on purpose: dividing by a rolling RV
    #     Gaussianizes any series mechanically, so its marginal cannot separate
    #     the models.
    def comove(x):
        xc = x - x.mean(axis=2, keepdims=True)
        sd = x.std(axis=2) + 1e-18
        cov = np.einsum("nit,njt->nij", xc, xc) / x.shape[2]
        cm = (cov / (sd[:, :, None] * sd[:, None, :])).mean(axis=0)
        return float(cm[~np.eye(x.shape[1], dtype=bool)].mean())

    ax = axes[0, 0]
    ks = [k for k in dec if k != "Real"]
    vals = [comove(dec[k][1]) for k in ks]
    rv = comove(dec["Real"][1])
    ax.bar(range(len(ks)), vals, 0.62, color=[C[k] for k in ks], lw=0)
    ax.axhline(rv, lw=1.1, color=C["Real"], zorder=4)
    ax.text(
        len(ks) - 0.5,
        rv,
        f" real {rv:.2f}",
        fontsize=5.2,
        va="bottom",
        ha="right",
        color=C["Real"],
    )
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.2f}", fontsize=5.0, ha="center", color=C[ks[i]])
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(
        [k.replace("Diffusion", "Diff.") for k in ks],
        rotation=32,
        fontsize=5.2,
        ha="right",
    )
    ax.set_ylim(0, max(max(vals), rv) * 1.25)
    ax.set_ylabel("mean off diagonal corr.", fontsize=6.4)
    ax.set_title(r"comovement of $x_t$ across assets", fontsize=7.0, pad=3)
    ax.grid(axis="y", alpha=0.5, lw=0.4)
    ax.set_axisbelow(True)
    tag(ax, "a", dx=-0.26, dy=1.10)

    # (b) the log volatility channel
    ax = axes[0, 1]
    g = np.linspace(-6, 6, 400)
    ax.plot(
        g,
        np.exp(-(g**2) / 2) / np.sqrt(2 * np.pi),
        lw=0.7,
        ls="--",
        color="#888888",
        zorder=1,
    )
    for k, (_, x) in dec.items():
        ax.hist(
            std(x),
            bins=120,
            range=(-6, 6),
            density=True,
            histtype="step",
            lw=1.2 if k == "Real" else 0.9,
            color=C[k],
            label=k,
            zorder=3 if k == "Real" else 2,
        )
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.0)
    ax.set_xlim(-6, 6)
    ax.set_title(r"log volatility $x_t = \log\sqrt{RV_t}$", fontsize=7.0, pad=3)
    tag(ax, "b", dx=-0.20, dy=1.10)

    # (c) clustering, measured in the volatility channel itself
    ax = axes[1, 0]
    lags = np.arange(1, NL + 1)
    for k, (_, x) in dec.items():
        ax.plot(
            lags,
            _acf_rows(x, NL).reshape(-1, NL).mean(axis=0),
            lw=1.2 if k == "Real" else 0.9,
            ls="-" if k == "Real" else "--",
            color=C[k],
            zorder=3 if k == "Real" else 2,
        )
    ax.axvline(10, lw=0.6, ls=":", color="#999999")
    ax.text(10.6, 0.93, "window\noverlap", fontsize=4.8, color="#777777", va="top")
    ax.axhline(0, lw=0.4, color="#AAAAAA")
    ax.set_ylim(-0.05, 1.02)
    ax.set_xlabel(r"lag $\tau$", fontsize=6.6)
    ax.set_ylabel("autocorrelation", fontsize=6.6)
    ax.set_title(r"$\mathrm{ACF}(x_t)$", fontsize=7.0, pad=3)
    tag(ax, "c", dx=-0.24, dy=1.10)

    # (d) whatever clustering is left in the shock channel is misplaced
    ax = axes[1, 1]
    for k, (z, _) in dec.items():
        ax.plot(
            lags,
            _acf_rows(z**2, NL).reshape(-1, NL).mean(axis=0),
            lw=1.2 if k == "Real" else 0.9,
            ls="-" if k == "Real" else "--",
            color=C[k],
            zorder=3 if k == "Real" else 2,
        )
    ax.axhline(0, lw=0.4, color="#AAAAAA")
    ax.set_ylim(-0.06, 0.10)
    ax.set_xlabel(r"lag $\tau$", fontsize=6.6)
    ax.set_title(r"$\mathrm{ACF}(z_t^2)$, residual", fontsize=7.0, pad=3)
    tag(ax, "d", dx=-0.20, dy=1.10)

    axes[0, 1].legend(
        loc="upper left",
        bbox_to_anchor=(-1.28, 1.46),
        ncol=3,
        fontsize=5.4,
        handlelength=1.1,
        columnspacing=0.8,
    )
    save(fig, "sq_8_decomposition_diagnostic.png")


# --------------------------------------------------------------------------


FIGURES = {
    1: sq1_radar_fidelity,
    2: sq2_radar_risk,
    3: sq3_radar_ablation,
    4: sq4_qq,
    5: sq5_tradeoff,
    6: sq6_corr_grid,
    7: sq7_ccf_lag1,
    8: sq8_decomposition_diagnostic,
}


if __name__ == "__main__":
    want = [int(a) for a in sys.argv[1:]] or sorted(FIGURES)
    for i in want:
        print(f"square figure {i} ...")
        FIGURES[i]()
    print("done.")
