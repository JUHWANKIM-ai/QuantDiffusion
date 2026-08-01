"""Publication PNG figures for the QuantDiffusion paper.

Seven self-contained candidates, all computed from this repository:

  data/indices.csv                                 real daily log returns
  data/RV/RV_Close_to_Close.csv                    realized variance proxy
  result/statistics/synthetic_data/<model>/*.npy   generated scenarios, 5 seeds
  result/statistics/*.csv                          the metric tables in the paper

Real data is the out of sample 30% tail, windowed the same way as in
2_benchmark.ipynb, so real and generated are compared window for window.

Run:  python make_figures.py            all figures
      python make_figures.py 3 5        only figures 3 and 5
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)  # repo root (utils/ is one level down)
HERE = os.path.join(
    ROOT, "result", "images", "figures"
)  # figures output dir (unchanged semantics)
os.makedirs(HERE, exist_ok=True)
SD = os.path.join(ROOT, "result", "statistics", "synthetic_data")
ST = os.path.join(ROOT, "result", "statistics")

SEQ_LEN = 256
RV_WINDOW = 10
EPS = 1e-8
DPI = 400
ASSETS = ["DJIA", "NASDAQ", "JPM", "HSI", "Gold", "WTI"]

# one colour per model, used identically in every figure
C = {
    "Real": "#111111",
    "QuantDiffusion": "#2B5C9B",
    "QuantGAN": "#C0651F",
    "TailGAN": "#4C8C40",
    "Diffusion-TS": "#7C5296",
    "Full": "#2B5C9B",
    "w/o RV": "#7C5296",
    "w/o Lambert W": "#C0651F",
    "w/o DMS-TCN": "#4C8C40",
    "w/o L1 loss": "#B25550",
    "w/o cosine": "#3E7C7B",
}
# The full model is cosine + smooth-L1 + Gaussian, the Full column of ablation2,
# whose scenarios live in a2_full; the tables read the same folders, so figures
# and tables cannot drift apart.
MODELS = {
    "QuantDiffusion": "baseline",
    "QuantGAN": "quantgan",
    "TailGAN": "tailgan",
    "Diffusion-TS": "diffusionTS",
}
ABL = {
    "Full": "baseline",
    "w/o RV": "a2_wo_rv",
    "w/o Lambert W": "a2_wo_lambertw",
    "w/o DMS-TCN": "a2_wo_moderntcn",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.4,
        "axes.labelsize": 7.4,
        "axes.titlesize": 8.0,
        "axes.linewidth": 0.7,
        "axes.edgecolor": "#555555",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "legend.fontsize": 6.6,
        "legend.frameon": False,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.5,
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def save(fig, name):
    p = os.path.join(HERE, name)
    fig.savefig(p, dpi=DPI, transparent=True)
    plt.close(fig)
    print("  wrote", os.path.relpath(p, ROOT))


def tag(ax, letter, dx=-0.10, dy=1.06):
    ax.text(
        dx,
        dy,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=8.4,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def rolling_window(a, w):
    n = a.shape[0] - w + 1
    return np.stack([a[i : i + w] for i in range(n)])


def load_real():
    """Out of sample real windows, shaped (N, 6, 256) like the generated files."""
    df = pd.read_csv(os.path.join(ROOT, "data", "indices.csv"))
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df = df.set_index("Date").apply(pd.to_numeric).astype(float)
    r = np.log1p(df.pct_change()).dropna()
    cut = int(0.7 * len(r))
    test = r.iloc[cut:, :].values
    return rolling_window(test, SEQ_LEN).transpose(0, 2, 1), r


def load_model(tag_, run=0):
    p = os.path.join(SD, tag_, f"{run}.npy")
    return np.load(p) if os.path.exists(p) else None


def acf_batch(X, nlags):
    """ACF up to nlags for every row of X (n, T)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    den = (Xc**2).sum(axis=1) + EPS
    out = np.empty((X.shape[0], nlags))
    for k in range(1, nlags + 1):
        out[:, k - 1] = (Xc[:, :-k] * Xc[:, k:]).sum(axis=1)
    return out / den[:, None]


def xcorr_batch(X, Y, nlags):
    """Corr(x_t, y_{t+tau}) for every row, used for the leverage effect."""
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    out = np.empty((X.shape[0], nlags))
    for k in range(1, nlags + 1):
        a, b = Xc[:, :-k], Yc[:, k:]
        out[:, k - 1] = (a * b).sum(axis=1) / (
            np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1)) + EPS
        )
    return out


def flat_windows(A, n_win=400):
    """(N, 6, 256) -> (n_win*6, 256), a stack of single asset series."""
    A = A[:n_win]
    return A.reshape(-1, A.shape[-1])


def excess_kurt(v):
    v = np.asarray(v, float).ravel()
    v = v[np.isfinite(v)]
    v = v - v.mean()
    s = v.std()
    return float((v**4).mean() / (s**4 + EPS) - 3.0) if s > 0 else np.nan


def horizon_returns(A, h, n_win=600):
    """Non overlapping h step sums inside each window, pooled over assets."""
    A = A[:n_win]
    m = (A.shape[-1] // h) * h
    return A[..., :m].reshape(A.shape[0], A.shape[1], -1, h).sum(-1).ravel()


def read_pm_table(path):
    """'0.0572 ± 0.0007' cells -> (means, stds) frames."""
    df = pd.read_csv(path, index_col=0)
    cell = df.map if hasattr(df, "map") else df.applymap  # pandas 2.1 renamed it
    mean = cell(lambda s: float(str(s).split("±")[0]))
    std = cell(lambda s: float(str(s).split("±")[1]) if "±" in str(s) else 0.0)
    return mean, std


def lambert_gaussianize(v):
    """Inverse Lambert W map with delta chosen so the excess kurtosis vanishes.

    If the input is already thin tailed there is no root, and the honest answer
    is delta = 0, i.e. the identity, rather than an arbitrary fallback that would
    compress an already Gaussian sample.
    """
    from scipy.optimize import brentq
    from scipy.special import lambertw

    v = np.asarray(v, float).ravel()
    u = (v - v.mean()) / (v.std() + EPS)

    def kurt(delta):
        w = np.sign(u) * np.sqrt(np.real(lambertw(delta * u**2)) / delta)
        w = (w - w.mean()) / (w.std() + EPS)
        return float((w**4).mean() - 3.0)

    if kurt(1e-6) <= 0:
        return u, 0.0
    try:
        delta = brentq(kurt, 1e-6, 3.0, xtol=1e-6)
    except ValueError:
        return u, 0.0
    w = np.sign(u) * np.sqrt(np.real(lambertw(delta * u**2)) / delta)
    return (w - w.mean()) / (w.std() + EPS), delta


def load_rv(mode="forward"):
    """Returns, realized volatility and the shock channel, aligned.

    mode='forward' reproduces the pipeline in 2_benchmark.ipynb, where RV_t
    averages days t to t+9 and therefore contains r_t itself. mode='backward'
    uses days t-10 to t-1 instead, so r_t is outside the window; it exists only
    to check that the conditional heteroskedasticity in figure 8 is not an
    artefact of that overlap.
    """
    df = pd.read_csv(os.path.join(ROOT, "data", "indices.csv"))
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df = df.set_index("Date").apply(pd.to_numeric).astype(float)
    r = np.log1p(df.pct_change()).dropna()
    rv = pd.read_csv(
        os.path.join(ROOT, "data", "RV", "RV_Close_to_Close.csv"), index_col=0
    ).rename_axis("Date")
    rv.index = pd.to_datetime(rv.index)
    for c in rv.columns:
        pos = rv[c][rv[c] > 0]
        rv[c] = rv[c].clip(lower=float(pos.quantile(0.01)) if len(pos) else 1e-8)
    idx = r.index.intersection(rv.index)
    r, rv = r.loc[idx], rv.loc[idx]
    roll = rv.rolling(RV_WINDOW, min_periods=min(5, RV_WINDOW)).mean()
    shifted = roll.shift(-(RV_WINDOW - 1)) if mode == "forward" else roll.shift(1)
    frv = np.sqrt(shifted)
    z = r / (frv + EPS)
    keep = pd.concat([z, np.log(frv + EPS)], axis=1).dropna().index
    return r.loc[keep], frv.loc[keep], z.loc[keep]


def pooled_rv_pairs(mode="forward"):
    """Pool the six assets into dimensionless (r, v) pairs.

    Each asset is divided by its own median realized variance: r -> r/sqrt(vbar)
    and v -> v/vbar. Since z = r/sqrt(v) is invariant under that rescaling and
    x = ln v only shifts, pooling costs nothing in the decomposed coordinates.
    """
    r, frv, _ = load_rv(mode)
    R, V = [], []
    for c in r.columns:
        v = frv[c].values ** 2
        rr = r[c].values
        ok = np.isfinite(v) & np.isfinite(rr) & (v > 0)
        v, rr = v[ok], rr[ok]
        vbar = float(np.median(v))
        V.append(v / vbar)
        R.append(rr / np.sqrt(vbar))
    R, V = np.concatenate(R), np.concatenate(V)
    return R, V, R / np.sqrt(V), np.log(V)


def sqrt_law_fit(R, V, nbins=16):
    """Regress std(r | v) on sqrt(v) through the origin over quantile bins."""
    edges = np.quantile(V, np.linspace(0, 1, nbins + 1))
    sd, sq, n = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (V >= lo) & (V < hi)
        if sel.sum() < 40:
            continue
        sd.append(R[sel].std())
        sq.append(np.sqrt(np.median(V[sel])))
        n.append(int(sel.sum()))
    sd, sq = np.array(sd), np.array(sq)
    c = float((sd * sq).sum() / (sq**2).sum())
    ss_res = float(((sd - c * sq) ** 2).sum())
    ss_tot = float(((sd - sd.mean()) ** 2).sum())
    return c, 1.0 - ss_res / (ss_tot + EPS), sd, sq, n


def hist_logy(ax, v, color, label, bins=160, rng=(-8, 8), step=True):
    v = np.asarray(v, float).ravel()
    v = v[np.isfinite(v)]
    v = (v - v.mean()) / (v.std() + EPS)
    kw = dict(bins=bins, range=rng, density=True, color=color, label=label)
    if step:
        ax.hist(v, histtype="step", lw=0.9, **kw)
    else:
        ax.hist(v, alpha=0.75, lw=0, **kw)
    ax.set_yscale("log")
    ax.set_xlim(*rng)


def gauss_ref(ax, rng=(-8, 8)):
    g = np.linspace(*rng, 500)
    ax.plot(
        g,
        np.exp(-(g**2) / 2) / np.sqrt(2 * np.pi),
        lw=0.8,
        ls="--",
        color="#777777",
        label=r"$\mathcal{N}(0,1)$",
    )


# ==========================================================================
# 1. energy landscape
# ==========================================================================
def fig1_landscape():
    _, frv, _ = load_rv()
    lv = np.log(frv["JPM"].dropna().values ** 2)
    m, s = lv.mean(), lv.std()
    vmax = float(np.exp(m + 2.2 * s))
    rmax = 3.2 * np.sqrt(vmax)

    rr = np.linspace(-rmax, rmax, 420)
    vv = np.linspace(vmax * 1e-3, vmax, 380)
    R, V = np.meshgrid(rr, vv)
    U1 = 0.5 * np.log(V) + R**2 / (2 * V) + 0.5 * ((np.log(V) - m) / s) ** 2
    zz = np.linspace(-3.2, 3.2, 420)
    xx = np.linspace(m - 2.2 * s, m + 2.2 * s, 380)
    Z, X = np.meshgrid(zz, xx)
    U2 = 0.5 * Z**2 + 0.5 * ((X - m) / s) ** 2

    fig = plt.figure(figsize=(7.0, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.34, 1], wspace=0.04)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])]
    mid = fig.add_subplot(gs[0, 1])
    mid.axis("off")

    for ax, (A, B, U, xl, yl, ti) in zip(
        axes,
        [
            (R, V, U1, "$r$", "$v$", r"raw coordinates:  $g(r,v)\propto\sqrt{v}$"),
            (Z, X, U2, "$z$", r"$x=\ln v$", r"decomposed coordinates:  $g(z)$"),
        ],
    ):
        U = U - U.min()
        lev = np.linspace(0, np.percentile(U, 92), 24)
        cf = ax.contourf(A, B, np.clip(U, lev[0], lev[-1]), levels=lev, cmap="Blues")
        ax.contour(
            A,
            B,
            np.clip(U, lev[0], lev[-1]),
            levels=lev[::3],
            colors="#33507A",
            linewidths=0.3,
            alpha=0.7,
        )
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(ti, pad=4)
        ax.set_xlim(A.min(), A.max())
        ax.set_ylim(B.min(), B.max())
        for sp in ax.spines.values():
            sp.set_visible(True)

    ax = axes[0]
    ax.plot(1.9 * np.sqrt(vv), vv, lw=0.8, color="#8C3B36", ls="--")
    ax.plot(-1.9 * np.sqrt(vv), vv, lw=0.8, color="#8C3B36", ls="--")
    ax.text(
        rmax * 0.42,
        vmax * 0.86,
        r"valley width $\propto\sqrt{v}$",
        fontsize=6.4,
        color="#8C3B36",
        ha="center",
        va="center",
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0),
    )
    ax.text(
        0,
        vmax * 0.05,
        "funnel: the Langevin step\nsize must track $v$",
        fontsize=6.4,
        color="#8C3B36",
        ha="center",
        va="bottom",
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0),
    )
    ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 2))
    ax.xaxis.get_offset_text().set_fontsize(5.8)
    ax.yaxis.get_offset_text().set_fontsize(5.8)
    tag(ax, "a")

    ax = axes[1]
    ax.yaxis.tick_right()  # keep the gutter free for the arrow
    ax.yaxis.set_label_position("right")
    for x0 in [m + 1.7 * s, m - 1.7 * s]:
        ax.annotate(
            "",
            xy=(0.15, m),
            xytext=(2.5, x0),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=0.7,
                color="#2F6B2A",
                connectionstyle="arc3,rad=0.12",
            ),
        )
    ax.text(
        0,
        m - 2.0 * s,
        "isotropic bowl: one step\nsize everywhere",
        fontsize=6.4,
        color="#2F6B2A",
        ha="center",
        va="bottom",
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0),
    )
    tag(ax, "b", dx=0.0)

    mid.annotate(
        "",
        xy=(0.95, 0.5),
        xytext=(0.05, 0.5),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#4C8C40"),
    )
    mid.text(
        0.5,
        0.585,
        r"$(z,x)=\left(\dfrac{r}{\sqrt{v}},\ \ln v\right)$",
        transform=mid.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.8,
        color="#2F6B2A",
        bbox=dict(fc="#EAF3E8", ec="#4C8C40", lw=0.6, pad=2.4),
    )
    # NOT Itô's lemma: this arrow is a Jacobian change of variables acting on a
    # static density. Itô's lemma acts on the SDE, which is what Section 3.2 of
    # the paper does with the same map.
    mid.text(
        0.5,
        0.435,
        "change of variables",
        transform=mid.transAxes,
        ha="center",
        va="top",
        fontsize=6.2,
        color="#555555",
        style="italic",
    )
    save(fig, "png_1_energy_landscape.png")


# ==========================================================================
# 2. what the decomposition does to the marginals
# ==========================================================================
def fig2_decomposition():
    """Two routes out of the heavy tail, and the scale process that drives one.

    Panel (b) applies the Lambert W map to the raw returns so its effect is
    visible in isolation. In the pipeline the map acts on the shock channel of
    panel (c), where, as the delta annotation shows, far less tail mass is left
    to attenuate.
    """
    r, frv, z = load_rv()
    # the Lambert map is fitted per channel, as in utils.scaling, then pooled
    lam_r, d_r = zip(*[lambert_gaussianize(r[c].values) for c in r.columns])
    lam_z, d_z = zip(*[lambert_gaussianize(z[c].values) for c in z.columns])
    lam_r = np.concatenate(lam_r)
    d_r, d_z = float(np.mean(d_r)), float(np.mean(d_z))

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95))
    fig.subplots_adjust(wspace=0.30)

    for ax, (v, ti, col) in zip(
        axes[:3],
        [
            (r.values, r"raw returns $r_t$", C["QuantDiffusion"]),
            (lam_r, r"Lambert $W^{-1}$ on $r_t$", "#C87F2A"),
            (z.values, r"$r_t/\sqrt{RV_t}$", "#4C8C40"),
        ],
    ):
        hist_logy(ax, v, col, None, bins=150, step=False)
        gauss_ref(ax)
        ax.set_title(ti, pad=4, fontsize=7.2)
        ax.set_ylim(2e-5, 1.0)
        ax.set_xticks([-5, 0, 5])
        ax.text(
            0.03,
            0.06,
            f"excess kurt.\n{excess_kurt(v):.2f}",
            transform=ax.transAxes,
            fontsize=6.2,
            va="bottom",
            ha="left",
            color=col,
        )
    axes[0].set_ylabel("density (log scale)")
    axes[1].text(
        0.97,
        0.06,
        rf"mean $\delta={d_r:.3f}$",
        transform=axes[1].transAxes,
        fontsize=6.2,
        va="bottom",
        ha="right",
        color="#C87F2A",
    )
    axes[2].text(
        0.97,
        0.06,
        rf"mean $\delta={d_z:.3f}$",
        transform=axes[2].transAxes,
        fontsize=6.2,
        va="bottom",
        ha="right",
        color="#4C8C40",
    )
    axes[0].legend(loc="upper right", handlelength=1.1)

    ax = axes[3]
    v = frv.values.ravel()
    v = v[np.isfinite(v)]
    ax.hist(
        v,
        bins=140,
        range=(0, np.quantile(v, 0.995)),
        density=True,
        color="#B25550",
        alpha=0.8,
        lw=0,
    )
    ax.set_title(r"scale process $\sqrt{RV_t}$", pad=4, fontsize=7.2)
    ax.set_xlabel(r"$\sqrt{RV_t}$")
    ax.set_yticks([])
    ax.xaxis.set_major_locator(MaxNLocator(4))

    for i, ax in enumerate(axes):
        tag(ax, "abcd"[i], dx=-0.20, dy=1.10)
    save(fig, "png_2_decomposition.png")


# ==========================================================================
# 3. stylized facts against the baselines
# ==========================================================================
def fig3_stylized_facts():
    real, _ = load_real()
    NL, NW = 40, 400
    data = {"Real": real}
    for name, tg in MODELS.items():
        a = load_model(tg)
        if a is not None:
            data[name] = a

    fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.2))
    fig.subplots_adjust(wspace=0.28, hspace=0.42)
    lags = np.arange(1, NL + 1)

    def curves(fn):
        out = {}
        for name, A in data.items():
            F = flat_windows(A, NW)
            out[name] = fn(F).mean(axis=0)
        return out

    panels = [
        (
            axes[0, 0],
            curves(lambda F: acf_batch(np.abs(F), NL)),
            r"$\mathrm{ACF}(|r_\tau|)$",
            "autocorrelation",
        ),
        (
            axes[0, 1],
            curves(lambda F: acf_batch(F**2, NL)),
            r"$\mathrm{ACF}(r_\tau^{2})$",
            None,
        ),
        (
            axes[1, 0],
            curves(lambda F: xcorr_batch(F, F**2, NL)),
            r"leverage effect  $\mathrm{Corr}(r_t, r_{t+\tau}^{2})$",
            "correlation",
        ),
    ]
    for ax, cur, ti, yl in panels:
        for name in ["Real"] + [k for k in data if k != "Real"]:
            ax.plot(
                lags,
                cur[name],
                lw=1.1 if name == "Real" else 0.9,
                ls="-" if name == "Real" else "--",
                color=C[name],
                label=name,
                zorder=3 if name == "Real" else 2,
            )
        ax.axhline(0, lw=0.5, color="#AAAAAA", zorder=1)
        ax.set_title(ti, pad=4)
        ax.set_xlabel(r"lag $\tau$")
        if yl:
            ax.set_ylabel(yl)
        ax.grid(axis="y", alpha=0.6)

    # Kurt(h) is read straight from the paper's metric table rather than
    # recomputed, so this panel cannot drift from Table 1. An independent
    # recomputation reproduces the ranking but not the exact estimator in
    # utils/metric.py, which is why the tabulated values are used here.
    ax = axes[1, 1]
    hs = [1, 5, 20, 100]
    km, ks_ = read_pm_table(os.path.join(ST, "benchmark__gen_quality.csv"))
    col = {
        "QuantDiffusion": "QuantDiffusion (ours, C2C/w10)",
        "QuantGAN": "QuantGAN",
        "TailGAN": "TailGAN",
        "Diffusion-TS": "Diffusion-TS",
    }
    width = 0.19
    names = [k for k in data if k != "Real"]
    for i, name in enumerate(names):
        rows = [f"Kurt({h})" for h in hs]
        m = [km.loc[r, col[name]] for r in rows]
        e = [ks_.loc[r, col[name]] for r in rows]
        ax.bar(
            np.arange(len(hs)) + (i - (len(names) - 1) / 2) * width,
            m,
            width,
            yerr=e,
            error_kw=dict(lw=0.6, capsize=1.2),
            color=C[name],
            label=name,
            lw=0,
        )
    ax.set_xticks(np.arange(len(hs)))
    ax.set_xticklabels([f"$h={h}$" for h in hs])
    ax.set_title(r"$\mathrm{Kurt}(h)$ error by horizon", pad=4)
    ax.set_ylabel(r"$|K(\hat r)-K(r)|$")
    ax.grid(axis="y", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", ncol=2, handlelength=1.1, columnspacing=0.8)

    axes[0, 0].legend(loc="upper right", ncol=1, handlelength=1.4)
    for ax, l in zip(axes.ravel(), "abcd"):
        tag(ax, l, dx=-0.13)
    save(fig, "png_3_stylized_facts.png")


# ==========================================================================
# 4. portfolio tail risk
# ==========================================================================
def fig4_tail_risk():
    real, _ = load_real()

    def pnl(A, n=1600):
        """Equal weight buy and hold P&L, prices starting at 100 per asset."""
        px = 100.0 * np.exp(np.cumsum(A[:n], axis=-1))
        return px.mean(axis=1)[:, -1] - 100.0

    series = {"Real": pnl(real)}
    for name, tg in MODELS.items():
        a = load_model(tg)
        if a is not None:
            series[name] = pnl(a)

    fig = plt.figure(figsize=(7.0, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.28, 1, 1], wspace=0.34)
    ax = fig.add_subplot(gs[0, 0])
    lo = min(np.quantile(v, 0.002) for v in series.values()) - 8
    hi = max(np.quantile(v, 0.995) for v in series.values()) + 8
    for name, v in series.items():
        ax.hist(
            np.clip(v, lo, hi),
            bins=100,
            range=(lo, hi),
            density=True,
            histtype="step",
            lw=1.2 if name == "Real" else 0.9,
            color=C[name],
            label=name,
            zorder=3 if name == "Real" else 2,
        )
    q = np.quantile(series["Real"], 0.05)
    ax.axvline(q, lw=0.8, ls=":", color="#B25550", zorder=1)
    ax.axvspan(lo, q, color="#F8CECC", alpha=0.35, lw=0, zorder=0)
    ax.set_yscale("log")
    ax.set_ylim(2e-4, 0.35)
    ax.set_xlim(lo, hi)
    ax.text(
        q - 3,
        0.22,
        "5% tail of\nthe real P&L",
        fontsize=5.8,
        color="#B25550",
        ha="right",
        va="top",
    )
    ax.set_xlabel("terminal P&L, equal weight portfolio")
    ax.set_ylabel("density (log scale)")
    ax.set_title("P&L distribution", pad=4)
    ax.legend(loc="upper right", handlelength=1.2, labelspacing=0.28)
    tag(ax, "a", dx=-0.14)

    mean, std = read_pm_table(os.path.join(ST, "benchmark__tailgan_vares.csv"))
    cols = {
        "QuantDiffusion (ours, C2C/w10)": "QuantDiffusion",
        "QuantGAN": "QuantGAN",
        "TailGAN": "TailGAN",
        "Diffusion-TS": "Diffusion-TS",
    }
    alphas = ["0.05", "0.025", "0.01"]
    for j, metric in enumerate(["VaR", "ES"]):
        ax = fig.add_subplot(gs[0, 1 + j])
        rows = [f"RE_{metric}({a})" for a in alphas]
        width = 0.2
        for i, (raw, name) in enumerate(cols.items()):
            m = [mean.loc[r, raw] for r in rows]
            e = [std.loc[r, raw] for r in rows]
            ax.bar(
                np.arange(3) + (i - 1.5) * width,
                m,
                width,
                yerr=e,
                error_kw=dict(lw=0.6, capsize=1.4),
                color=C[name],
                label=name,
                lw=0,
            )
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels([rf"$\alpha={a}$" for a in alphas])
        ax.set_title(rf"relative error of {metric}", pad=4)
        if j == 0:
            ax.set_ylabel("relative error")
        ax.set_ylim(0, 0.95)
        ax.grid(axis="y", alpha=0.6)
        ax.set_axisbelow(True)
        tag(ax, "bc"[j], dx=-0.16)
    save(fig, "png_4_tail_risk.png")


# ==========================================================================
# 5. ablation
# ==========================================================================
def fig5_ablation():
    gm, gs_ = read_pm_table(os.path.join(ST, "ablation__gen_quality.csv"))
    tm, ts = read_pm_table(os.path.join(ST, "ablation__tailgan_vares.csv"))
    raw = {
        "Full": "Baseline",
        "w/o RV": "w/o RV Decomp.",
        "w/o Lambert W": "w/o LambertW",
        "w/o DMS-TCN": "w/o DMS-TCN",
    }

    left = ["ACF(|x|)", "ACF(x^2)", "Leverage Effect"]
    right = ["RE_VaR(0.05)", "RE_ES(0.05)", "RE_VaR(0.01)", "RE_ES(0.01)"]

    fig, axes = plt.subplots(
        1, 2, figsize=(7.0, 2.35), gridspec_kw=dict(width_ratios=[1, 1.28], wspace=0.30)
    )
    width = 0.2
    for ax, rows, mm, ss, ti, yl in [
        (axes[0], left, gm, gs_, "temporal dependence", "metric (lower is better)"),
        (axes[1], right, tm, ts, "portfolio tail risk", "relative error"),
    ]:
        for i, (name, col) in enumerate(raw.items()):
            m = [mm.loc[r, col] for r in rows]
            e = [ss.loc[r, col] for r in rows]
            ax.bar(
                np.arange(len(rows)) + (i - 1.5) * width,
                m,
                width,
                yerr=e,
                error_kw=dict(lw=0.6, capsize=1.4),
                color=C[name],
                label=name,
                lw=0,
            )
        ax.set_xticks(np.arange(len(rows)))
        ax.set_xticklabels(
            [
                r.replace("ACF(", "ACF(")
                .replace("Leverage Effect", "leverage")
                .replace("RE_", "")
                .replace("x^2", "$x^2$")
                for r in rows
            ],
            fontsize=6.4,
        )
        ax.set_title(ti, pad=4)
        ax.set_ylabel(yl)
        ax.grid(axis="y", alpha=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper left", ncol=2, handlelength=1.2, columnspacing=0.9)
    axes[0].set_ylim(0, 0.0052)
    axes[1].set_ylim(0, 0.78)
    for ax, l in zip(axes, "ab"):
        tag(ax, l, dx=-0.14)
    save(fig, "png_5_ablation.png")


# ==========================================================================
# 6. noise structure
# ==========================================================================
def fig6_noise_structure():
    from scipy.stats import t as tdist

    def cosine_alphabar(T, tau):
        t = np.linspace(0, 1, T + 1)[1:]
        v0, v1 = 1.0, np.cos(np.pi / 2) ** (2 * tau)
        g = np.cos(t * np.pi / 2) ** (2 * tau)
        return np.clip(1.0 - (v0 - g) / (v0 - v1 + 1e-12), 1e-9, 1.0)

    T = 75
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    fig.subplots_adjust(wspace=0.34)

    ax = axes[0]
    g = np.linspace(-7, 7, 700)
    nu = 5.0
    sc = np.sqrt((nu - 2) / nu)
    ax.plot(
        g,
        np.exp(-(g**2) / 2) / np.sqrt(2 * np.pi),
        lw=0.9,
        ls="--",
        color="#777777",
        label=r"$\mathcal{N}(0,1)$",
    )
    ax.plot(
        g,
        tdist.pdf(g / sc, df=nu) / sc,
        lw=1.1,
        color="#7C5296",
        label=r"$t_5$, unit variance",
    )
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 0.7)
    ax.set_xlim(-7, 7)
    ax.set_ylabel("density (log scale)")
    ax.set_title("prior noise distribution", pad=4)
    ax.legend(loc="lower center", handlelength=1.3)

    ax = axes[1]
    for tv, col, lw in [
        (0.5, "#CCCCCC", 0.8),
        (1.0, "#AAAAAA", 0.8),
        (1.5, "#7C5296", 1.3),
        (2.0, "#888888", 0.8),
    ]:
        ax.plot(
            np.arange(1, T + 1),
            cosine_alphabar(T, tv),
            lw=lw,
            color=col,
            label=rf"$\tau={tv}$" + (" (used)" if tv == 1.5 else ""),
        )
    ax.set_xlabel("diffusion step $t$")
    ax.set_ylabel(r"$\bar\alpha_t$")
    ax.set_title(r"cosine schedule", pad=4)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="lower left", handlelength=1.3)

    ax = axes[2]
    ab = cosine_alphabar(T, 1.5)
    sig, noi = np.sqrt(ab), np.sqrt(1 - ab)
    tg = np.arange(1, T + 1)
    ax.plot(tg, sig, lw=1.2, color="#2B5C9B")
    ax.plot(tg, noi, lw=1.2, color="#B25550")
    cross = int(np.argmin(np.abs(sig - noi))) + 1
    ax.axvspan(cross, T, color="#F8CECC", alpha=0.35, lw=0)
    ax.axvline(cross, lw=0.6, ls=":", color="#888888")
    ax.text(
        4,
        0.90,
        r"$\sqrt{\bar\alpha_t}$  (signal)",
        fontsize=6.4,
        color="#2B5C9B",
        ha="left",
        va="center",
    )
    ax.text(
        T - 2,
        0.86,
        r"$\sqrt{1-\bar\alpha_t}$" + "\n(injected noise)",
        fontsize=6.4,
        color="#B25550",
        ha="right",
        va="top",
    )
    ax.text(
        (cross + T) / 2,
        0.34,
        "the stochastic volatility\nBrownian term is\nindistinguishable here",
        fontsize=5.9,
        color="#8C3B36",
        ha="center",
        va="center",
    )
    ax.set_xlabel("diffusion step $t$")
    ax.set_ylabel("amplitude")
    ax.set_title("signal against injected noise", pad=4)
    ax.set_ylim(0, 1.04)

    for ax, l in zip(axes, "abc"):
        tag(ax, l, dx=-0.17)
    save(fig, "png_6_noise_structure.png")


# ==========================================================================
# 7. cross asset dependence
# ==========================================================================
def fig7_correlation():
    real, _ = load_real()

    def corr(A, n=1200):
        F = A[:n].transpose(0, 2, 1).reshape(-1, A.shape[1])
        return np.corrcoef(F, rowvar=False)

    mats = {"Real": corr(real)}
    for name, tg in MODELS.items():
        a = load_model(tg)
        if a is not None:
            mats[name] = corr(a)

    # the annotated score is the paper's own CCF(tau = 0) MAE, read from the
    # metric table, so the figure cannot disagree with Table 1
    ccf, _ = read_pm_table(os.path.join(ST, "benchmark__gen_quality.csv"))
    ccf_col = {
        "QuantDiffusion": "QuantDiffusion (ours, C2C/w10)",
        "QuantGAN": "QuantGAN",
        "TailGAN": "TailGAN",
        "Diffusion-TS": "Diffusion-TS",
    }

    order = ["Real", "QuantDiffusion", "Diffusion-TS", "QuantGAN", "TailGAN"]
    order = [o for o in order if o in mats]
    fig, axes = plt.subplots(1, len(order), figsize=(7.0, 1.8))
    fig.subplots_adjust(wspace=0.12)
    for ax, name in zip(axes, order):
        M = mats[name]
        im = ax.imshow(M, vmin=-0.3, vmax=1.0, cmap="RdBu_r")
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels(ASSETS, rotation=90, fontsize=5.2)
        ax.set_yticklabels(ASSETS if name == order[0] else [], fontsize=5.2)
        err = (
            ""
            if name == "Real"
            else f"\nCCF($\\tau$=0) {ccf.loc['CCF(τ = 0)_MAE', ccf_col[name]]:.4f}"
        )
        ax.set_title(
            name + err,
            fontsize=6.8,
            pad=3,
            color=C[name],
            fontweight="bold" if name == "Real" else "normal",
        )
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.012)
    cb.ax.tick_params(labelsize=5.6, length=1.6)
    cb.outline.set_linewidth(0.4)
    save(fig, "png_7_correlation.png")


# ==========================================================================
# 8. energy landscape, fully empirical
# ==========================================================================
def fig8_landscape_empirical():
    """Figure 1 with the model replaced by a 2D histogram of the actual pairs."""
    from scipy.ndimage import gaussian_filter

    R, V, Z, X = pooled_rv_pairs("forward")
    c_f, r2_f, sd, sq, nb = sqrt_law_fit(R, V)
    Rb, Vb, _, _ = pooled_rv_pairs("backward")
    c_b, r2_b, *_ = sqrt_law_fit(Rb, Vb)
    print(f"  n = {R.size:,} pooled pairs")
    print(
        f"  forward  window (contains r_t): std(r|v) = {c_f:.3f} sqrt(v), "
        f"R2 = {r2_f:.3f}"
    )
    print(
        f"  backward window (excludes r_t): std(r|v) = {c_b:.3f} sqrt(v), "
        f"R2 = {r2_b:.3f}"
    )

    def surface(A, B, arng, brng, bins, min_count=4):
        H, ae, be = np.histogram2d(A, B, bins=bins, range=[arng, brng])
        dens = gaussian_filter(H, 0.9)
        dens /= dens.sum() * (ae[1] - ae[0]) * (be[1] - be[0])
        U = -np.log(np.maximum(dens, 1e-12))
        U = U - U.min()
        return np.ma.masked_where(gaussian_filter(H, 0.9) < min_count, U).T, ae, be

    # empirical conditional 5% and 95% quantiles: the funnel, measured
    def cond_q(A, B, edges, min_n=60):
        mid_, qlo, qhi = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (B >= lo) & (B < hi)
            if sel.sum() < min_n:
                continue
            mid_.append(np.median(B[sel]))
            q = np.quantile(A[sel], [0.05, 0.95])
            qlo.append(q[0])
            qhi.append(q[1])
        return np.array(mid_), np.array(qlo), np.array(qhi)

    # size the axes from the quantile envelope so the data fills the frame
    v_hi = float(np.quantile(V, 0.97))
    vm, qlo, qhi = cond_q(R, V, np.quantile(V[V <= v_hi], np.linspace(0, 1, 17)))
    r_hi = 1.32 * float(max(qhi.max(), -qlo.min()))
    x_lo, x_hi = np.quantile(X, [0.004, 0.996])
    xm, zlo, zhi = cond_q(Z, X, np.linspace(x_lo, x_hi, 17))
    z_hi = 1.32 * float(max(zhi.max(), -zlo.min()))

    U1, ae1, be1 = surface(R, V, (-r_hi, r_hi), (0, v_hi), (58, 44))
    U2, ae2, be2 = surface(Z, X, (-z_hi, z_hi), (x_lo, x_hi), (58, 44))

    fig = plt.figure(figsize=(7.0, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.34, 1], wspace=0.04)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])]
    mid = fig.add_subplot(gs[0, 1])
    mid.axis("off")

    for ax, (U, ae, be, xl, yl, ti) in zip(
        axes,
        [
            (
                U1,
                ae1,
                be1,
                r"$r\,/\,\sqrt{\bar v}$",
                r"$v\,/\,\bar v$",
                r"empirical $-\log\hat p(r,v)$",
            ),
            (
                U2,
                ae2,
                be2,
                "$z$",
                r"$x=\ln (v/\bar v)$",
                r"empirical $-\log\hat p(z,x)$",
            ),
        ],
    ):
        lev = np.linspace(0, np.percentile(U.compressed(), 95), 22)
        ax.contourf(
            0.5 * (ae[:-1] + ae[1:]),
            0.5 * (be[:-1] + be[1:]),
            np.clip(U, lev[0], lev[-1]),
            levels=lev,
            cmap="Blues",
        )
        ax.set_facecolor("white")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(ti, pad=4)
        ax.set_xlim(ae[0], ae[-1])
        ax.set_ylim(be[0], be[-1])
        for sp in ax.spines.values():
            sp.set_visible(True)

    ax = axes[0]
    ax.plot(qlo, vm, lw=1.1, color="#8C3B36")
    ax.plot(qhi, vm, lw=1.1, color="#8C3B36")
    vg = np.linspace(vm.min(), vm.max(), 200)  # clipped to the data range
    ax.plot(1.645 * c_f * np.sqrt(vg), vg, lw=0.9, ls="--", color="#222222")
    ax.plot(-1.645 * c_f * np.sqrt(vg), vg, lw=0.9, ls="--", color="#222222")
    ax.text(
        0.5,
        0.97,
        "solid: empirical 5% and 95%\n"
        rf"dashed: $\pm1.645\,c\sqrt{{v}}$,  $c={c_f:.2f}$,  $R^2={r2_f:.2f}$"
        + "\n"
        + rf"$c={c_b:.2f}$ also with a backward window",
        transform=ax.transAxes,
        fontsize=5.4,
        color="#8C3B36",
        ha="center",
        va="top",
        bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.4),
    )
    ax.text(
        0.5,
        0.035,
        "the spread widens with $v$",
        transform=ax.transAxes,
        fontsize=6.4,
        color="#8C3B36",
        ha="center",
        va="bottom",
        bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.0),
    )
    tag(ax, "a")

    ax = axes[1]
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.plot(zlo, xm, lw=1.1, color="#2F6B2A")
    ax.plot(zhi, xm, lw=1.1, color="#2F6B2A")
    spread = zhi - zlo
    ax.text(
        0.5,
        0.035,
        "the spread no longer depends on $x$:\n"
        rf"the 5% to 95% width varies by "
        rf"{100 * spread.std() / spread.mean():.0f}% across $x$",
        transform=ax.transAxes,
        fontsize=6.0,
        color="#2F6B2A",
        ha="center",
        va="bottom",
        bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.2),
    )
    tag(ax, "b", dx=0.0)

    mid.annotate(
        "",
        xy=(0.95, 0.5),
        xytext=(0.05, 0.5),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#4C8C40"),
    )
    mid.text(
        0.5,
        0.585,
        r"$z=\dfrac{r}{\sqrt{v}}$",
        transform=mid.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#2F6B2A",
        bbox=dict(fc="#EAF3E8", ec="#4C8C40", lw=0.6, pad=2.4),
    )
    mid.text(
        0.5,
        0.435,
        f"same {R.size // 1000}k observations",
        transform=mid.transAxes,
        ha="center",
        va="top",
        fontsize=6.2,
        color="#555555",
        style="italic",
    )
    save(fig, "png_8_energy_landscape_empirical.png")


FIGURES = {
    1: fig1_landscape,
    2: fig2_decomposition,
    3: fig3_stylized_facts,
    4: fig4_tail_risk,
    5: fig5_ablation,
    6: fig6_noise_structure,
    7: fig7_correlation,
    8: fig8_landscape_empirical,
}


if __name__ == "__main__":
    want = [int(a) for a in sys.argv[1:]] or sorted(FIGURES)
    for i in want:
        print(f"figure {i} ...")
        FIGURES[i]()
    print("done.")
