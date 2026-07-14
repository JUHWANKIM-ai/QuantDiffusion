"""
QuantGAN benchmark runner (unified convention).

Running `python run.py` end-to-end:
  1. reads the shared dataset  data/indices.csv  (repo root),
  2. trains QuantGAN (TCN generator/discriminator),
  3. picks the best epoch by 100-day cumulative EMD (no look-ahead: vs train data),
  4. saves the best generator + 5 synthetic sets of 2000 samples to
        result/statistics/synthetic_data/quantgan/
            0.npy ... 4.npy        # each (2000, 6, 256) float32
            generator_best.pt      # best-epoch generator weight

This matches the diffusionTS / TailGAN runners so all three benchmarks share
the same output layout.

    python run.py                 # full pipeline
    python run.py --epochs 50     # short test run
"""

import argparse
import os
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
os.chdir(HERE)                       # so `from src...`, ./results/ resolve correctly
sys.path.insert(0, HERE)

from src.utils import scaling, inverse_scaling
from src.baselines.networks.generators import UserGenerator
from src.baselines.networks.discriminators import UserDiscriminator
from src.baselines.trainer import GANTrainer

DATA_CSV = os.path.join(PROJECT, "data", "indices.csv")      # shared dataset (repo root)
RESULT_DIR = os.path.join(PROJECT, "result", "statistics", "synthetic_data", "quantgan")
WEIGHT_DIR = os.path.join(PROJECT, "result", "weight", "quantgan")   # transient per-epoch checkpoints
SEQ_LEN = 256        # unified benchmark length (each synthetic series is (6, 256))
N_SETS = 5           # number of synthetic sets (0.npy ... 4.npy)
SET_SIZE = 2000      # samples per set


def rolling_window(data, window_size):
    n_windows = data.shape[0] - window_size + 1
    windows = np.zeros((n_windows, window_size, data.shape[1]))
    for idx in range(n_windows):
        windows[idx] = data[idx:idx + window_size]
    return windows


def compute_avg_emd(real_data, fake_data, window):
    """Earth Mover's Distance of `window`-day cumulative returns, summed over features."""
    emd = 0
    for i in range(real_data.shape[1]):
        real_dist = rolling_window(real_data[:, i, :].T, window).sum(axis=1).ravel()
        fake_dist = rolling_window(fake_data[:, i, :].T, window).sum(axis=1).ravel()
        emd += wasserstein_distance(real_dist, fake_dist)
    return emd


def make_config(epochs, device):
    return SimpleNamespace(
        file_name="indices", data_dir="data/", result_dir="results/", log_interval=5,
        n_vars=6, n_steps=SEQ_LEN, device=device, seed=0,
        batch_size=256, n_epochs=epochs, algo="QuantGAN",
        G_hidden_dim=80, D_hidden_dim=80, output_dim=1, noise_dim=6, clip=0.01,
        lr_G=0.0002, lr_D=0.0001, D_steps_per_G_step=1,
    )


def load_data(config):
    df = pd.read_csv(DATA_CSV)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df.set_index("Date", inplace=True)
    df = df.apply(pd.to_numeric).astype(float)

    log_returns = np.log1p(df.pct_change()).dropna().values
    train_raw = log_returns[:int(0.7 * len(log_returns))]      # time-ordered split

    train_scaled, scalers = scaling(train_raw)                 # (T, 6) gaussianized
    train_windowed = torch.from_numpy(rolling_window(train_scaled, SEQ_LEN)).float()  # (B, T, 6)
    training_set = TensorDataset(train_windowed)
    train_dl = DataLoader(training_set, batch_size=config.batch_size, shuffle=False)
    print(f"[data] train windows: {tuple(train_windowed.shape)}")
    return train_dl, train_windowed, scalers


def select_best_epoch(config, scalers, training_data):
    """Load each saved epoch and keep the one with lowest EMD vs the training data."""
    models_dir = WEIGHT_DIR   # 중간 체크포인트는 result/weight/quantgan 에 저장됨

    real_data = inverse_scaling(training_data.transpose(1, 2), scalers)
    generator = UserGenerator(config).to(config.device)

    best_epoch, min_dist = 0, float("inf")
    for epoch in range(50, config.n_epochs, 5):
        ckpt = f"{models_dir}/Generator_{epoch}.pt"
        if not os.path.exists(ckpt):
            continue
        generator.load_state_dict(torch.load(ckpt, map_location=config.device))
        generator.eval()
        with torch.no_grad():
            noise = torch.randn(100, config.noise_dim, config.n_steps, device=config.device)
            fake = generator(noise)
        emd = compute_avg_emd(real_data, inverse_scaling(fake, scalers), window=100)
        if emd < min_dist:
            min_dist, best_epoch = emd, epoch
            print(f"[select] epoch={best_epoch}  EMD={min_dist:.4f}")

    best_ckpt = f"{models_dir}/Generator_{best_epoch}.pt"
    generator.load_state_dict(torch.load(best_ckpt, map_location=config.device))
    generator.eval()
    print(f"[select] best epoch = {best_epoch}")
    return generator


def generate_sets(generator, scalers, config, n_sets):
    """Generate n_sets synthetic sets of 2000 -> quantgan/{0..n_sets-1}.npy."""
    os.makedirs(RESULT_DIR, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        for s in range(n_sets):
            noise = torch.randn(SET_SIZE, config.noise_dim, config.n_steps, device=config.device)
            fake = generator(noise)
            fake_data = inverse_scaling(fake, scalers).astype(np.float32)   # (2000, 6, 256)
            out_path = os.path.join(RESULT_DIR, f"{s}.npy")
            np.save(out_path, fake_data)
            print(f"[gen] set {s}: shape={fake_data.shape} -> {out_path}")


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
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--train", action="store_true", help="새로 훈련 (미지정 시 대화형 질문)")
    ap.add_argument("--no-train", action="store_true", help="훈련 없이 기존 weight으로 생성만")
    ap.add_argument("--sets", type=int, default=None, help="생성할 2000개 묶음 수 (미지정 시 질문)")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested but CUDA is not available.")

    do_train, n_sets = ask_plan(args)
    config = make_config(args.epochs, args.device)
    train_dl, training_data, scalers = load_data(config)
    weight_path = os.path.join(RESULT_DIR, "generator_best.pt")

    if do_train:
        generator = UserGenerator(config)
        discriminator = UserDiscriminator(config)
        trainer = GANTrainer(G=generator, D=discriminator, train_dl=train_dl, config=config)
        # 중간 체크포인트는 result/weight/quantgan 에 저장 (모듈 내 ./results 사용 안 함)
        shutil.rmtree(WEIGHT_DIR, ignore_errors=True)
        os.makedirs(WEIGHT_DIR, exist_ok=True)
        trainer.results_dir = WEIGHT_DIR
        shutil.rmtree(os.path.join(HERE, "results"), ignore_errors=True)  # 생성자가 만든 빈 폴더 정리
        print(f"[run] training QuantGAN for {config.n_epochs} epochs...")
        trainer.fit()                                                   # result/weight/quantgan/Generator_{e}.pt
        generator = select_best_epoch(config, scalers, training_data)   # best epoch by EMD
        os.makedirs(RESULT_DIR, exist_ok=True)
        torch.save(generator.state_dict(), weight_path)
        print(f"[run] generator weight saved: {weight_path}")
        shutil.rmtree(WEIGHT_DIR, ignore_errors=True)                   # 중간 체크포인트 삭제
        print(f"[run] removed intermediate checkpoints: {WEIGHT_DIR}")
    else:
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"기존 weight 없음: {weight_path} — 먼저 훈련(--train)하세요.")
        generator = UserGenerator(config).to(config.device)
        generator.load_state_dict(torch.load(weight_path, map_location=config.device))
        print(f"[run] 기존 weight 로드: {weight_path}")

    print(f"[run] 2000개 x {n_sets}묶음 생성...")
    generate_sets(generator, scalers, config, n_sets)


if __name__ == "__main__":
    main()
