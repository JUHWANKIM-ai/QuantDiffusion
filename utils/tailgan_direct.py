"""Direct-read variant of TailGAN's Compute_PNL.

TG.Compute_PNL reads the static-portfolio transition matrix from a fixed file (the live slot
data/TailGAN/.../LShort_Stk6/TransMat_{IS,OOS}.npy), so evaluating many baskets requires copying
each basket into that slot. This module replicates Compute_PNL exactly but takes the transition
matrix as an ARGUMENT, so the eval can read a basket path directly with no file swap.

Only the static-portfolio branch touches the basket; BuyHold / MeanRev / TrendFollow act on the
raw asset prices and are unchanged. Verified bit-for-bit against the copy method in
scratch/verify_direct.py.

    import tailgan_eval as TE
    TG = TE.configure_from_project(ROOT)
    from utils.tailgan_direct import compute_pnl_direct, vares_relerror_direct
    d = vares_relerror_direct(TG, real, fake, np.load(basket_oos_path), (0.05, 0.025, 0.01))
"""
import numpy as np
import torch


def compute_pnl_direct(TG, R, transmat):
    """TG.Compute_PNL(R) with StaticPort fed `transmat` (np array (n_stocks, K)) directly."""
    opt = TG.opt
    prices_l = TG.Inc2Price(R)
    # ---- StaticPort inline (Transform.StaticPort L57-65), transmat passed instead of file-read ----
    tm = TG.Tensor(np.asarray(transmat, dtype=np.float32)[:, :opt.n_trans])
    swap_prices = prices_l.permute(0, 2, 1)
    broad = tm.reshape(1, *tm.shape).repeat(swap_prices.size(0), 1, 1)
    port_prices_l = torch.bmm(swap_prices, broad).permute(0, 2, 1)
    # ---- identical strategy P&L stack as Compute_PNL ----
    PNL_l = [TG.BuyHold(prices_l, opt.Cap)]
    for strategy in opt.strategies:
        if strategy == "Port":
            PNL_l.append(TG.BuyHold(port_prices_l, opt.Cap))
        elif strategy == "MR":
            for percentile_l in opt.thresholds_pct:
                th = TG.gen_thresholds(opt.data_name, opt.tickers, strategy, percentile_l, 100, opt.WH)
                PNL_l.append(TG.MeanRev(prices_l, opt.Cap, opt.WH, LR=opt.ratios[0], SR=opt.ratios[1],
                                        ST=th[:, -1], LT=th[:, -2]))
        elif strategy == "TF":
            for percentile_l in opt.thresholds_pct:
                th = TG.gen_thresholds(opt.data_name, opt.tickers, strategy, percentile_l, 100, opt.WH)
                PNL_l.append(TG.TrendFollow(prices_l, opt.Cap, opt.WH, LR=opt.ratios[0], SR=opt.ratios[1],
                                            ST=th[:, 0], LT=th[:, 1]))
    return torch.cat(PNL_l, dim=1)


def vares_relerror_direct(TG, real_bct, fake_bct, transmat, alphas=(0.05, 0.025, 0.01)):
    """VaR/ES relative error (fake vs real) using compute_pnl_direct; no file swap."""
    import tailgan_eval as TE
    rp = compute_pnl_direct(TG, TG.Tensor(np.asarray(real_bct, dtype=np.float32)), transmat).cpu().detach().numpy()
    fp = compute_pnl_direct(TG, TG.Tensor(np.asarray(fake_bct, dtype=np.float32)), transmat).cpu().detach().numpy()
    return TE.relerror(rp, fp, alphas)
