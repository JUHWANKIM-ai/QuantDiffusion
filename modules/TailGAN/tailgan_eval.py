"""
Ported TailGAN VaR/ES evaluation (original-paper metric).

This reuses the ORIGINAL TailGAN strategy machinery (Compute_PNL: BuyHold + EigenPort
static portfolios + Mean-Reversion + Trend-Following) and the paper's Empirical_Stats
(empirical VaR = percentile, ES = mean beyond VaR) to score a model by the RELATIVE
error of its VaR/ES estimates vs the real data — the exact quantity TailGAN optimizes.

Two entry points:
  * configure_from_project(...)  -> imports/sets up the flat TailGAN modules (TG, Transform,
    gen_thresholds, gen_static_port) against data/TailGAN, ensures the static-portfolio
    transition matrix exists, and returns the TailGAN module (use TG.Compute_PNL / TG.Tensor).
  * vares_relerror(compute_pnl, tensor_fn, real_bct, fake_bct, alphas) -> {"RE_VaR(a)":..,"RE_ES(a)":..}

Used by BOTH modules/TailGAN/run.py (--select vares, epoch selection on train data) and
3_benchmark.ipynb (per-model ported metric). It is ADDITIVE — the existing eval_portfolio
RE_VaR/RE_ES metric is unchanged.
"""

import os
import shutil
import sys
from os.path import join

import numpy as np

DEFAULT_ALPHAS = (0.05, 0.025, 0.01)

# The basket that every reported VaR/ES number is measured against, tracked in the
# repository so a fresh clone reproduces the tables. See ensure_static_basket.
REFERENCE_BASKET = join("data", "tailgan_basket")


def ensure_static_basket(project_root, gsp, n_rows=6, n_ports=50,
                         static_way="LShort", seed=0):
    """Put the static portfolio matrix in place without ever redrawing it.

    Those n_ports portfolios ARE the evaluation basket: 50 of Compute_PNL's 68
    columns, so every relative error in the result tables is measured against
    them. Gen_StaticPort draws them from an unseeded normal through the global
    numpy state, and run.py used to call it on every --train. Redrawing rescores
    every model at once, including models that were not retrained: on 2026-07-28
    a redraw moved QuantGAN's RE_VaR(0.05) from 0.5340 to 0.4468 and
    Diffusion-TS's from 0.5684 to 0.6333 from byte identical samples. The two
    tables were not comparable and neither was reproducible.

    A seed alone does not fix this, because data/TailGAN is not in the repository,
    so a fresh clone would draw its own basket, and because scipy.sparse.random
    and stats.norm.rvs are not guaranteed stable across scipy versions. The basket
    is therefore checked in under data/tailgan_basket and copied into place here.

    Order of preference: an existing working copy, then the tracked reference,
    then a seeded draw as a last resort. To move to a new basket deliberately,
    replace the tracked reference and expect every number to change.
    """
    live_dir = join(project_root, "data", "TailGAN", "Static_Port_Transform",
                    f"{static_way}_Stk{n_rows}")
    ref_dir = join(project_root, REFERENCE_BASKET, f"{static_way}_Stk{n_rows}")
    live = join(live_dir, "TransMat_IS.npy")
    if os.path.exists(live):
        return live
    if os.path.exists(join(ref_dir, "TransMat_IS.npy")):
        os.makedirs(live_dir, exist_ok=True)
        for f in ("TransMat_IS.npy", "TransMat_OOS.npy"):
            src = join(ref_dir, f)
            if os.path.exists(src):
                shutil.copyfile(src, join(live_dir, f))
        print(f"[basket] restored from the tracked reference: {ref_dir}")
        return live
    np.random.seed(seed)                     # Gen_StaticPort draws from this state
    gsp.Gen_StaticPort(static_way, n_stocks=n_rows, n_ports=n_ports)
    print(f"[basket] no tracked reference found, drew a new basket with seed "
          f"{seed}. Every number in the tables will differ from the reported "
          f"ones; copy this matrix to {ref_dir} to freeze it.")
    return live


def empirical_var_es(pnl, alpha):
    """Empirical VaR/ES per PnL column (matches TailGAN Evaluation.Empirical_Stats).

    pnl: (n_samples, n_cols). Returns (var[n_cols], es[n_cols]) for a left-tail alpha (<0.5).
    """
    var = np.percentile(pnl, alpha * 100, axis=0)               # (n_cols,)
    mask = pnl <= var.reshape(1, -1)                            # samples in the tail
    tail = np.where(mask, pnl, np.nan)
    with np.errstate(invalid="ignore"):
        es = np.nanmean(tail, axis=0)                          # (n_cols,)
    return var, es


def relerror(real_pnl, fake_pnl, alphas=DEFAULT_ALPHAS):
    """Mean relative error |real - fake| / |real| of VaR/ES over all PnL columns, per alpha."""
    eps = 1e-8
    out = {}
    for a in alphas:
        rv, re_ = empirical_var_es(real_pnl, a)
        fv, fe = empirical_var_es(fake_pnl, a)
        var_re = np.nanmean(np.abs(rv - fv) / (np.abs(rv) + eps))
        es_re = np.nanmean(np.abs(re_ - fe) / (np.abs(re_) + eps))
        out[f"RE_VaR({a})"] = float(var_re)
        out[f"RE_ES({a})"] = float(es_re)
    return out


def vares_relerror(compute_pnl, tensor_fn, real_bct, fake_bct, alphas=DEFAULT_ALPHAS):
    """Score fake vs real (both (N, n_rows, T) returns) by TailGAN VaR/ES relative error.

    compute_pnl : TG.Compute_PNL   (returns torch PnL tensor (N, n_strategy_cols))
    tensor_fn   : TG.Tensor        (numpy/torch -> device FloatTensor)
    """
    real_pnl = compute_pnl(tensor_fn(np.asarray(real_bct, dtype=np.float32))).cpu().detach().numpy()
    fake_pnl = compute_pnl(tensor_fn(np.asarray(fake_bct, dtype=np.float32))).cpu().detach().numpy()
    return relerror(real_pnl, fake_pnl, alphas)


def configure_from_project(project_root, data_name="indices2_6d",
                           tickers=("DJI", "NASDAQ", "JPM", "HSI", "GOLD", "WTI_OIL"),
                           n_rows=6, n_cols=256, n_trans=50, static_way="LShort",
                           alphas=DEFAULT_ALPHAS):
    """Set up the flat TailGAN modules against data/TailGAN and return the TailGAN module.

    Must be called from a context where modules/TailGAN is importable. Ensures the static
    portfolio transition matrix exists (creates it if missing). Assumes gan_data windows +
    threshold caches already exist from a prior `run.py --train` (they do after retraining).
    """
    HERE = os.path.dirname(os.path.abspath(__file__))
    YOUR_PATH = join(project_root, "data", "TailGAN")
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    _cwd = os.getcwd()
    os.chdir(HERE)
    sys.argv = ["TailGAN.py"]                      # keep TailGAN's argparse defaults
    try:
        import TailGAN as TG
        import Transform as tf
        import gen_thresholds as gt
        import gen_static_port as gsp

        # paths
        TG.your_path = YOUR_PATH
        tf.your_path = YOUR_PATH
        tf.trans_parent_data_path = join(YOUR_PATH, "Static_Port_Transform")
        gt.your_path = YOUR_PATH
        gt.parent_data_path = join(YOUR_PATH, "gan_data")
        gsp.your_path = YOUR_PATH
        gsp.trans_parent_data_path = join(YOUR_PATH, "Static_Port_Transform")

        # opt fields used by Compute_PNL / strategies
        o = TG.opt
        o.n_rows, o.n_cols = n_rows, n_cols
        o.tickers = list(tickers)
        o.data_name = data_name
        o.static_way = static_way
        o.n_trans = n_trans
        o.alphas = list(alphas)
        TG.R_shape = (n_rows, n_cols)

        # put the evaluation basket in place from the tracked reference
        ensure_static_basket(project_root, gsp, n_rows=n_rows, n_ports=n_trans,
                             static_way=static_way)
        return TG
    finally:
        os.chdir(_cwd)
