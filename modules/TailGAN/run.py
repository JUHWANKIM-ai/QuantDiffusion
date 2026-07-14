"""
TailGAN benchmark runner (unified convention).

Running `python run.py` end-to-end:
  1. reads the shared dataset  data/indices.csv  (repo root) and builds the
     rolling-window CSVs TailGAN expects,
  2. trains an ENSEMBLE of --ensemble models (original TailGAN numNN, default 10;
     distinct seeds), original TailGAN.py logic, saving generators every 100 epochs,
  3. selects, per the ORIGINAL-PAPER ensemble scheme: screens models by final
     training-loss (keep <= --screen-pct percentile), then picks each survivor's
     best epoch by --select (default 'vares' = min VaR/ES relative error vs train
     data, the quantity TailGAN optimizes; also 'last' / 'emd'),
  4. pools samples equally across the surviving generators into 5 synthetic sets to
        result/statistics/synthetic_data/tailgan/
            0.npy ... 4.npy               # each (2000, 6, 256) float32
            generator_best_id{m}.pt       # best epoch of each surviving model

This matches the diffusionTS / QuantGAN runners so all three benchmarks share
the same output layout.

    python run.py                 # full pipeline
    python run.py --epochs 50     # short test run
"""

import argparse
import glob
import os
import re
import shutil
import sys
from os.path import join

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TG_DIR = join(PROJECT, "modules", "TailGAN")
YOUR_PATH = join(PROJECT, "data", "TailGAN")
DATA_CSV = join(PROJECT, "data", "indices.csv")           # shared dataset (repo root)
RESULT_DIR = join(PROJECT, "result", "statistics", "synthetic_data", "tailgan")
WEIGHT_DIR = join(PROJECT, "result", "weight", "tailgan")   # transient per-epoch checkpoints

TICKERS = ["DJI", "NASDAQ", "JPM", "HSI", "GOLD", "WTI_OIL"]
DATA_NAME = "indices2_6d"
N_ROWS, N_COLS = 6, 256
TRAIN_RATIO = 0.7
N_SETS = 5            # number of synthetic sets (0.npy ... 4.npy)
SET_SIZE = 2000       # samples per set


def build_windows():
    """data/indices.csv -> data/TailGAN/gan_data/indices2_6d/{i}.csv (one rolling window each)."""
    df = pd.read_csv(DATA_CSV)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df = df.set_index("Date").apply(pd.to_numeric).astype(float)
    tickers = list(df.columns)

    log_ret = np.log1p(df.pct_change()).dropna()
    train = log_ret.iloc[:int(TRAIN_RATIO * len(log_ret))]   # train split only (no test leak)

    out_dir = join(YOUR_PATH, "gan_data", DATA_NAME)
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        if f.endswith(".csv"):
            os.remove(join(out_dir, f))

    vals = train.values
    n_windows = len(vals) - N_COLS + 1
    for i in range(n_windows):
        pd.DataFrame(vals[i:i + N_COLS], columns=tickers).to_csv(
            join(out_dir, "%d.csv" % (i + 1)), index=False)
    print(f"[prep] {n_windows} windows -> {out_dir}  (tickers={tickers})")
    return n_windows


def train_windows_bct():
    """Train-split log-return rolling windows as (N, 6, 256) — EMD reference."""
    df = pd.read_csv(DATA_CSV)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df = df.set_index("Date").apply(pd.to_numeric).astype(float)
    log_ret = np.log1p(df.pct_change()).dropna().values
    vals = log_ret[:int(TRAIN_RATIO * len(log_ret))]
    n = len(vals) - N_COLS + 1
    out = np.zeros((n, N_COLS, vals.shape[1]), dtype=np.float32)
    for i in range(n):
        out[i] = vals[i:i + N_COLS]
    return np.transpose(out, (0, 2, 1)).astype(np.float32)  # (N, 6, 256)


def rolling_window(data, window_size):
    n = data.shape[0] - window_size + 1
    out = np.zeros((n, window_size, data.shape[1]), dtype=np.float64)
    for i in range(n):
        out[i] = data[i:i + window_size]
    return out


def compute_avg_emd(real_bct, fake_bct, window=100):
    """EMD of `window`-day cumulative returns, summed over the 6 features."""
    emd = 0.0
    for i in range(real_bct.shape[1]):
        real_dist = rolling_window(real_bct[:, i, :].T, window).sum(axis=1).ravel()
        fake_dist = rolling_window(fake_bct[:, i, :].T, window).sum(axis=1).ravel()
        emd += wasserstein_distance(real_dist, fake_dist)
    return emd


def _tail_noise(TG, n):
    """Student-t(5) (or normal) latent noise, matching the original TailGAN."""
    nm = TG.opt.noise_name
    if "t" in nm:
        z = np.random.standard_t(int(nm.split("t")[1]), (n, TG.opt.latent_dim))
    else:
        z = np.random.normal(0, 1, (n, TG.opt.latent_dim))
    return TG.Tensor(z)


def select_best_epoch(TG, real_bct, ckpt_dir, sample_n, min_epoch=2000):
    """Pick the lowest train-EMD checkpoint among epochs >= min_epoch (per-100-epoch saves)."""
    import torch

    def _ep(p):
        return int(re.search(r"_E(\d+)", p).group(1))

    all_ckpts = sorted(glob.glob(join(ckpt_dir, "generator_E*.pt")), key=_ep)
    if not all_ckpts:
        raise RuntimeError("no generator checkpoints were saved during training")
    ckpts = [c for c in all_ckpts if _ep(c) >= min_epoch]
    if not ckpts:
        print(f"[select] epoch>={min_epoch} 체크포인트 없음 -> 전체 epoch 에서 선택")
        ckpts = all_ckpts

    G = TG.Generator()
    if TG.cuda:
        G.cuda()
    best, best_emd = ckpts[-1], float("inf")
    print(f"[select] evaluating {len(ckpts)} checkpoints by train EMD (sample_n={sample_n})...")
    for ck in ckpts:
        ep = int(re.search(r"_E(\d+)", ck).group(1))
        G.load_state_dict(torch.load(ck, map_location="cuda" if TG.cuda else "cpu"))
        G.eval()
        with torch.no_grad():
            fake = G(_tail_noise(TG, sample_n)).cpu().numpy().astype(np.float32)
        emd = compute_avg_emd(real_bct, fake, 100)
        mark = ""
        if emd < best_emd:
            best_emd, best = emd, ck
            mark = "  <- best"
        print(f"[select] E{ep}  EMD={emd:.5f}{mark}")
    print(f"[select] best = {os.path.basename(best)}  EMD={best_emd:.5f}")
    return best


def select_last_epoch(ckpt_dir):
    """Use the final (highest-epoch) generator checkpoint, no selection."""
    def _ep(p):
        return int(re.search(r"_E(\d+)", p).group(1))

    all_ckpts = sorted(glob.glob(join(ckpt_dir, "generator_E*.pt")), key=_ep)
    if not all_ckpts:
        raise RuntimeError("no generator checkpoints were saved during training")
    last = all_ckpts[-1]
    print(f"[select] last weight = {os.path.basename(last)}")
    return last


def select_by_vares(TG, real_bct, ckpt_dir, sample_n, alphas=(0.05, 0.025, 0.01)):
    """Original TailGAN method: pick the checkpoint minimizing VaR/ES relative error vs train data.

    Uses the paper's strategy PnL machinery (TG.Compute_PNL: BuyHold+Port+MR+TF) + empirical
    VaR/ES, exactly the quantity TailGAN's discriminator optimizes. Evaluates the per-100-epoch
    generators saved under run.py's checkpoint dir (numNN=1, so no ensemble screening).
    """
    import torch
    import tailgan_eval

    def _ep(p):
        return int(re.search(r"_E(\d+)", p).group(1))

    ckpts = sorted(glob.glob(join(ckpt_dir, "generator_E*.pt")), key=_ep)
    if not ckpts:
        raise RuntimeError("no generator checkpoints were saved during training")

    real_pnl = TG.Compute_PNL(TG.Tensor(real_bct)).cpu().detach().numpy()   # (N, n_strat_cols), computed once
    G = TG.Generator()
    if TG.cuda:
        G.cuda()
    best, best_err = ckpts[-1], float("inf")
    print(f"[select] evaluating {len(ckpts)} checkpoints by TailGAN VaR/ES rel-error (sample_n={sample_n})...")
    for ck in ckpts:
        ep = _ep(ck)
        G.load_state_dict(torch.load(ck, map_location="cuda" if TG.cuda else "cpu"))
        G.eval()
        with torch.no_grad():
            fake = G(_tail_noise(TG, sample_n)).cpu().numpy().astype(np.float32)
        fake_pnl = TG.Compute_PNL(TG.Tensor(fake)).cpu().detach().numpy()
        d = tailgan_eval.relerror(real_pnl, fake_pnl, alphas)
        err = float(np.mean(list(d.values())))
        mark = ""
        if err < best_err:
            best_err, best = err, ck
            mark = "  <- best"
        print(f"[select] E{ep}  VaR/ES-RE={err:.5f}{mark}")
    print(f"[select] best (min VaR/ES-RE) = {os.path.basename(best)}  VaR/ES-RE={best_err:.5f}")
    return best


def screen_ensemble(gen_data_path, num_nn, thres_pct=50):
    """TailGAN convergence screen: keep models whose final generator loss <= thres_pct percentile.

    Based on training loss only (no look-forward bias), as in the original Screen_Ensemble.
    """
    finals = []
    for m in range(num_nn):
        loss = np.load(join(gen_data_path, f"loss_id{m}.npy"))   # (2, n_epochs): rows = [D, G]
        finals.append(float(loss[1, -1]))                        # final G loss
    thr = float(np.percentile(finals, thres_pct))
    survivors = [m for m in range(num_nn) if finals[m] <= thr]
    print("[screen] final G-loss: " + ", ".join(f"m{m}={v:.4f}" for m, v in enumerate(finals)))
    print(f"[screen] threshold(<= p{thres_pct})={thr:.4f} -> survivors {survivors}")
    return survivors


def _pick_epoch(TG, ckpts, metric, real_bct, real_pnl, sample_n, alphas=(0.05, 0.025, 0.01)):
    """Pick one checkpoint from a single model's saved epochs by `metric` (vares/emd/last)."""
    import torch
    import tailgan_eval

    if metric == "last":
        return ckpts[-1], float("nan")

    G = TG.Generator()
    if TG.cuda:
        G.cuda()
    best_p, best_v = ckpts[-1], float("inf")
    for ck in ckpts:
        G.load_state_dict(torch.load(ck, map_location="cuda" if TG.cuda else "cpu"))
        G.eval()
        with torch.no_grad():
            fake = G(_tail_noise(TG, sample_n)).cpu().numpy().astype(np.float32)
        if metric == "vares":
            fake_pnl = TG.Compute_PNL(TG.Tensor(fake)).cpu().detach().numpy()
            v = float(np.mean(list(tailgan_eval.relerror(real_pnl, fake_pnl, alphas).values())))
        else:  # emd
            v = compute_avg_emd(real_bct, fake, 100)
        if v < best_v:
            best_v, best_p = v, ck
    return best_p, best_v


def select_ensemble(TG, real_bct, ckpt_dir, sample_n, num_nn, screen_pct, gen_data_path,
                    metric="vares", alphas=(0.05, 0.025, 0.01)):
    """Original TailGAN ensemble selection: screen by convergence, then pick each survivor's
    best epoch by `metric`. Returns {model_index: best_ckpt_path}. (num_nn==1 -> single model.)
    """
    def _ep(p):
        return int(re.search(r"_E(\d+)", p).group(1))

    survivors = screen_ensemble(gen_data_path, num_nn, screen_pct) if num_nn > 1 else [0]
    real_pnl = (TG.Compute_PNL(TG.Tensor(real_bct)).cpu().detach().numpy()
                if metric == "vares" else None)
    best = {}
    for m in survivors:
        ckpts = sorted(glob.glob(join(ckpt_dir, f"generator_id{m}_E*.pt")), key=_ep)
        if not ckpts:
            print(f"[select] model {m}: no checkpoints, skip")
            continue
        bp, bv = _pick_epoch(TG, ckpts, metric, real_bct, real_pnl, sample_n, alphas)
        best[m] = bp
        print(f"[select] model {m}: best {os.path.basename(bp)}  ({metric}={bv:.5f})")
    if not best:
        raise RuntimeError("ensemble selection produced no generators")
    return best


def generate_ensemble(TG, weight_paths, n_sets, set_size=SET_SIZE):
    """Pool samples equally across the ensemble's best generators into n_sets of set_size."""
    import torch

    gens = []
    for p in weight_paths:
        g = TG.Generator()
        if TG.cuda:
            g.cuda()
        g.load_state_dict(torch.load(p, map_location="cuda" if TG.cuda else "cpu"))
        g.eval()
        gens.append(g)
    k = len(gens)
    per = int(np.ceil(set_size / k))
    os.makedirs(RESULT_DIR, exist_ok=True)
    with torch.no_grad():
        for s in range(n_sets):
            parts = [g(_tail_noise(TG, per)).cpu().numpy().astype(np.float32) for g in gens]
            pooled = np.concatenate(parts, axis=0)
            idx = np.random.permutation(pooled.shape[0])[:set_size]
            fake_data = pooled[idx]
            out_path = join(RESULT_DIR, f"{s}.npy")
            np.save(out_path, fake_data)
            print(f"[gen] set {s}: shape={fake_data.shape} (pooled from {k} model(s)) -> {out_path}")


def ask_plan(args):
    """Resolve (do_train, n_sets) from CLI flags, else interactive prompts."""
    if args.train:
        do_train = True
    elif args.no_train:
        do_train = False
    else:
        do_train = input("새로 훈련하시겠습니까? 기존 weight으로 생성만 하려면 n [y/N]: ").strip().lower() in ("y", "yes")
    n_sets = args.sets
    if n_sets is None:
        raw = input(f"2000개짜리 샘플을 몇 묶음 생성할까요? [기본 {N_SETS}]: ").strip()
        n_sets = int(raw) if (raw.isdigit() and int(raw) > 0) else N_SETS
    return do_train, n_sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("TG_EPOCHS", "20000")))
    ap.add_argument("--batch", type=int, default=None, help="batch size (default: TailGAN default)")
    ap.add_argument("--train", action="store_true", help="새로 훈련 (미지정 시 대화형 질문)")
    ap.add_argument("--no-train", action="store_true", help="훈련 없이 기존 weight으로 생성만")
    ap.add_argument("--sets", type=int, default=None, help="생성할 2000개 묶음 수 (미지정 시 질문)")
    ap.add_argument("--select-samples", type=int, default=1000,
                    help="best epoch EMD 선택 시 체크포인트당 생성 샘플 수")
    ap.add_argument("--select-min-epoch", type=int, default=2000,
                    help="best EMD 선택 후보의 최소 epoch (이 이상에서만 고름; --select=emd 일 때만)")
    ap.add_argument("--select", type=str, default="vares", choices=["vares", "last", "emd"],
                    help="모델별 epoch 선정: 'vares'=VaR/ES 상대오차 최소(원논문 기본), 'last'=마지막, 'emd'=train 최소 EMD")
    ap.add_argument("--ensemble", type=int, default=10, help="앙상블 모델 수 numNN (원논문 기본 10; 1이면 단일)")
    ap.add_argument("--screen-pct", type=int, default=50, help="수렴 스크리닝 백분위 (최종 G-loss 이하 생존)")
    args = ap.parse_args()

    do_train, n_sets = ask_plan(args)

    # TailGAN uses flat imports (from Dataset import ...) and relative paths -> run from its dir.
    sys.path.insert(0, TG_DIR)
    os.chdir(TG_DIR)
    sys.argv = ["TailGAN.py"]                       # let argparse keep its defaults
    os.makedirs(RESULT_DIR, exist_ok=True)

    if do_train:
        n_windows = build_windows()

        # ---- static-portfolio matrices + path injection (training only) ----
        import gen_static_port as gsp
        gsp.your_path = YOUR_PATH
        gsp.trans_parent_data_path = join(YOUR_PATH, "Static_Port_Transform")
        os.makedirs(gsp.trans_parent_data_path, exist_ok=True)
        gsp.Gen_StaticPort("LShort", n_stocks=N_ROWS, n_ports=50)

        import gen_thresholds as gt
        gt.your_path = YOUR_PATH
        gt.parent_data_path = join(YOUR_PATH, "gan_data")

        import Transform as tf
        tf.your_path = YOUR_PATH
        tf.trans_parent_data_path = join(YOUR_PATH, "Static_Port_Transform")

        import TailGAN as TG
        TG.your_path = YOUR_PATH
        o = TG.opt
        o.n_rows, o.n_cols = N_ROWS, N_COLS
        o.tickers = TICKERS
        o.data_name = DATA_NAME
        o.numNN = args.ensemble
        o.n_epochs = args.epochs
        if args.batch is not None:
            o.batch_size = args.batch
        # DataLoader has no drop_last; trim len to a multiple of batch so the last
        # partial batch can't mismatch Discriminator(Linear(batch_size)).
        o.len = (n_windows // o.batch_size) * o.batch_size
        TG.R_shape = (o.n_rows, o.n_cols)
        TG.PNL_shape = TG.Infer_Shape(TG.R_shape)
        TG.gen_data_path = join(YOUR_PATH, "Gens", DATA_NAME)
        TG.model_path = join(YOUR_PATH, "Models", DATA_NAME)
        os.makedirs(TG.gen_data_path, exist_ok=True)
        os.makedirs(TG.model_path, exist_ok=True)
        print(f"[run] R_shape={TG.R_shape}, len={o.len}, n_epochs={o.n_epochs}, batch={o.batch_size}")
        shutil.rmtree(join(TG_DIR, "your_path"), ignore_errors=True)  # clean stray import-time dir

        # ---- keep every per-100-epoch generator under result/weight (for EMD selection) ----
        tmp_ckpt = WEIGHT_DIR
        shutil.rmtree(tmp_ckpt, ignore_errors=True)
        os.makedirs(tmp_ckpt, exist_ok=True)

        import numpy as _np
        import torch as _torch
        _orig_np_save, _orig_torch_save = _np.save, _torch.save

        def _np_save_skip_fake(file, arr, *a, **k):
            if "Fake_id" in str(file):
                return
            return _orig_np_save(file, arr, *a, **k)

        def _torch_save_gen_only(obj, f, *a, **k):
            fs = str(f)
            if "generator" in fs:                      # keep per-(model, epoch) generator, drop discriminator
                mid = re.search(r"id(\d+)", fs)
                mep = re.search(r"_E(\d+)", fs)
                mm = int(mid.group(1)) if mid else 0
                ep = int(mep.group(1)) if mep else 0
                return _orig_torch_save(obj, join(tmp_ckpt, f"generator_id{mm}_E{ep}.pt"), *a, **k)
            return None

        _np.save, _torch.save = _np_save_skip_fake, _torch_save_gen_only
        # ---- train the ensemble (numNN models, distinct seeds), saving generators every 100 epochs ----
        from Dataset import Dataset_IS
        base_seed = getattr(TG, "seed", 1)
        dataset = Dataset_IS(tickers=o.tickers, data_path=join(YOUR_PATH, "gan_data", o.data_name), length=o.len)
        dataloader = _torch.utils.data.DataLoader(dataset, batch_size=o.batch_size, shuffle=True)
        for m in range(o.numNN):
            print(f"------ Model {m}/{o.numNN} (seed {base_seed + m}) ------")
            TG.Train_Single(o, dataloader, model_index=m, seed=base_seed + m)
        _np.save, _torch.save = _orig_np_save, _orig_torch_save

        # ---- original TailGAN ensemble: screen by convergence, pick each survivor's best epoch ----
        real_bct = train_windows_bct()
        best = select_ensemble(TG, real_bct, tmp_ckpt, args.select_samples, o.numNN,
                               args.screen_pct, TG.gen_data_path, metric=args.select)
        for fp in glob.glob(join(RESULT_DIR, "generator_best*.pt")):
            os.remove(fp)
        os.makedirs(RESULT_DIR, exist_ok=True)
        for m, p in best.items():
            shutil.copyfile(p, join(RESULT_DIR, f"generator_best_id{m}.pt"))
        shutil.rmtree(tmp_ckpt, ignore_errors=True)
        print(f"[run] ensemble: saved {len(best)} best generator(s) (select={args.select}, numNN={o.numNN})")
        weight_paths = sorted(glob.glob(join(RESULT_DIR, "generator_best_id*.pt")))
    else:
        weight_paths = sorted(glob.glob(join(RESULT_DIR, "generator_best_id*.pt")))
        if not weight_paths:
            raise FileNotFoundError(f"기존 weight 없음: {RESULT_DIR}/generator_best_id*.pt — 먼저 훈련(--train)하세요.")
        import TailGAN as TG                           # generation needs only the Generator class
        TG.your_path = YOUR_PATH
        TG.opt.n_rows, TG.opt.n_cols = N_ROWS, N_COLS
        TG.R_shape = (N_ROWS, N_COLS)
        shutil.rmtree(join(TG_DIR, "your_path"), ignore_errors=True)
        print(f"[run] 기존 앙상블 weight {len(weight_paths)}개 로드")

    print(f"[run] 2000개 x {n_sets}묶음 생성 (앙상블 {len(weight_paths)}개 pooling)...")
    generate_ensemble(TG, weight_paths, n_sets)


if __name__ == "__main__":
    main()
