"""
TailGAN benchmark runner (unified convention).

Running `python run.py` end-to-end:
  1. reads the shared dataset  data/indices.csv  (repo root) and builds the
     rolling-window CSVs TailGAN expects,
  2. trains TailGAN (original TailGAN.py logic, untouched),
  3. picks the per-100-epoch checkpoint with the lowest train-set EMD as the
     final generator (mirroring QuantGAN/diffusionTS),
  4. saves that generator + 5 synthetic sets of 2000 samples to
        result/statistics/synthetic_data/tailgan/
            0.npy ... 4.npy        # each (2000, 6, 256) float32
            generator_best.pt      # lowest train-EMD epoch

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
        print(f"[select] ⚠️ epoch>={min_epoch} 체크포인트 없음 -> 전체 epoch 에서 선택")
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
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("TG_EPOCHS", "3000")))
    ap.add_argument("--batch", type=int, default=None, help="batch size (default: TailGAN default)")
    ap.add_argument("--train", action="store_true", help="새로 훈련 (미지정 시 대화형 질문)")
    ap.add_argument("--no-train", action="store_true", help="훈련 없이 기존 weight으로 생성만")
    ap.add_argument("--sets", type=int, default=None, help="생성할 2000개 묶음 수 (미지정 시 질문)")
    ap.add_argument("--select-samples", type=int, default=1000,
                    help="best epoch EMD 선택 시 체크포인트당 생성 샘플 수")
    ap.add_argument("--select-min-epoch", type=int, default=2000,
                    help="best EMD 선택 후보의 최소 epoch (이 이상에서만 고름)")
    args = ap.parse_args()

    do_train, n_sets = ask_plan(args)

    # TailGAN uses flat imports (from Dataset import ...) and relative paths -> run from its dir.
    sys.path.insert(0, TG_DIR)
    os.chdir(TG_DIR)
    sys.argv = ["TailGAN.py"]                       # let argparse keep its defaults
    os.makedirs(RESULT_DIR, exist_ok=True)
    gen_final = join(RESULT_DIR, "generator_best.pt")

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
        o.numNN = 1
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
            if "generator" in fs:
                m = re.search(r"_E(\d+)", fs)
                ep = int(m.group(1)) if m else 0
                return _orig_torch_save(obj, join(tmp_ckpt, f"generator_E{ep}.pt"), *a, **k)
            return None  # discriminator/중간본은 저장 안 함

        _np.save, _torch.save = _np_save_skip_fake, _torch_save_gen_only
        TG.Train(o)                                    # train (saves generator every 100 epochs)
        _np.save, _torch.save = _orig_np_save, _orig_torch_save

        # ---- pick the lowest train-EMD epoch as the final generator ----
        real_bct = train_windows_bct()
        best_ckpt = select_best_epoch(TG, real_bct, tmp_ckpt, args.select_samples,
                                      min_epoch=args.select_min_epoch)
        shutil.copyfile(best_ckpt, gen_final)
        shutil.rmtree(tmp_ckpt, ignore_errors=True)
        print(f"[run] best (min train-EMD) generator saved: {gen_final}")
    else:
        if not os.path.exists(gen_final):
            raise FileNotFoundError(f"기존 weight 없음: {gen_final} — 먼저 훈련(--train)하세요.")
        import TailGAN as TG                           # generation needs only the Generator class
        TG.your_path = YOUR_PATH
        TG.opt.n_rows, TG.opt.n_cols = N_ROWS, N_COLS
        TG.R_shape = (N_ROWS, N_COLS)
        shutil.rmtree(join(TG_DIR, "your_path"), ignore_errors=True)
        print(f"[run] 기존 weight 로드: {gen_final}")

    print(f"[run] 2000개 x {n_sets}묶음 생성...")
    generate(TG, gen_final, n_sets)


def generate(TG, weight_path, n_sets):
    import torch

    generator = TG.Generator()
    if TG.cuda:
        generator.cuda()
    generator.load_state_dict(torch.load(weight_path, map_location="cuda" if TG.cuda else "cpu"))
    generator.eval()

    os.makedirs(RESULT_DIR, exist_ok=True)

    with torch.no_grad():
        for s in range(n_sets):
            gen_R = generator(_tail_noise(TG, SET_SIZE)).cpu().numpy().astype(np.float32)  # (2000, 6, 256)
            out_path = join(RESULT_DIR, f"{s}.npy")
            np.save(out_path, gen_R)
            print(f"[gen] set {s}: shape={gen_R.shape} -> {out_path}")


if __name__ == "__main__":
    main()
