"""
QuantGAN benchmark runner (unified 5-seed convention).

Running `python run.py` end-to-end:
  1. reads the shared dataset  data/indices.csv  (repo root),
  2. trains QuantGAN (TCN generator/discriminator) once per seed for seeds 0..4
     (5 independent models, all four RNGs pinned per seed),
  3. picks each seed's best epoch by 100-day cumulative EMD (no look-ahead: vs train data),
  4. saves, per seed, the best generator + ONE synthetic set of 2000 samples to
        result/statistics/synthetic_data/quantgan/
            0.npy ... 4.npy                 # one batch per model, each (2000, 6, 256) float32
            generator_best_seed{s}.pt       # best-epoch generator weight of seed s

One model -> one batch (5 models -> 5 batches), matching the quantDiffusion / TailGAN /
diffusionTS runners so every model follows the same 5-seed scheme.

    python run.py                 # full pipeline (5 seeds)
    python run.py --epochs 50     # short test run
    python run.py --no-train      # regenerate batches from existing per-seed weights
"""

import argparse
import os
import random
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
os.chdir(HERE)  # so `from src...`, ./results/ resolve correctly
sys.path.insert(0, HERE)

from src.utils import scaling, inverse_scaling
from src.baselines.networks.generators import UserGenerator
from src.baselines.networks.discriminators import UserDiscriminator
from src.baselines.trainer import GANTrainer

DATA_CSV = os.path.join(PROJECT, "data", "indices.csv")  # shared dataset (repo root)
RESULT_DIR = os.path.join(PROJECT, "result", "statistics", "synthetic_data", "quantgan")
WEIGHT_DIR = os.path.join(
    PROJECT, "result", "weight", "quantgan"
)  # transient per-epoch checkpoints
SEQ_LEN = 256  # unified benchmark length (each synthetic series is (6, 256))
SEEDS = [0, 1, 2, 3, 4]  # 5 models, one batch each
GEN_SEED_OFFSET = (
    1000  # generation reseeds at GEN_SEED_OFFSET + seed, separate from training
)
SET_SIZE = 2000  # samples per set (one set per seed)


def set_all_seeds(seed):
    """Pin all four RNGs (python / numpy / torch / cuda) so each seed is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rolling_window(data, window_size):
    n_windows = data.shape[0] - window_size + 1
    windows = np.zeros((n_windows, window_size, data.shape[1]))
    for idx in range(n_windows):
        windows[idx] = data[idx : idx + window_size]
    return windows


def compute_avg_emd(real_data, fake_data, window):
    """Earth Mover's Distance of `window`-day cumulative returns, summed over features."""
    emd = 0
    for i in range(real_data.shape[1]):
        real_dist = rolling_window(real_data[:, i, :].T, window).sum(axis=1).ravel()
        fake_dist = rolling_window(fake_data[:, i, :].T, window).sum(axis=1).ravel()
        emd += wasserstein_distance(real_dist, fake_dist)
    return emd


def make_config(epochs, device, seed=0):
    return SimpleNamespace(
        file_name="indices",
        data_dir="data/",
        result_dir="results/",
        log_interval=5,
        n_vars=6,
        n_steps=SEQ_LEN,
        device=device,
        seed=seed,
        batch_size=256,
        n_epochs=epochs,
        algo="QuantGAN",
        G_hidden_dim=80,
        D_hidden_dim=80,
        output_dim=1,
        noise_dim=6,
        clip=0.01,
        lr_G=0.0002,
        lr_D=0.0001,
        D_steps_per_G_step=1,
    )


def load_data(config):
    df = pd.read_csv(DATA_CSV)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df.set_index("Date", inplace=True)
    df = df.apply(pd.to_numeric).astype(float)

    log_returns = np.log1p(df.pct_change()).dropna().values
    train_raw = log_returns[: int(0.7 * len(log_returns))]  # time-ordered split

    train_scaled, scalers = scaling(train_raw)  # (T, 6) gaussianized
    train_windowed = torch.from_numpy(
        rolling_window(train_scaled, SEQ_LEN)
    ).float()  # (B, T, 6)
    training_set = TensorDataset(train_windowed)
    train_dl = DataLoader(training_set, batch_size=config.batch_size, shuffle=False)
    print(f"[data] train windows: {tuple(train_windowed.shape)}")
    return train_dl, train_windowed, scalers


def select_best_epoch(config, scalers, training_data, models_dir):
    """Load each saved epoch and keep the one with lowest EMD vs the training data."""
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
            noise = torch.randn(
                100, config.noise_dim, config.n_steps, device=config.device
            )
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


def generate_one(generator, scalers, config, seed, out_path):
    """Generate ONE synthetic set of 2000 for this seed -> quantgan/{seed}.npy.
    Reseeds at GEN_SEED_OFFSET + seed so generation is separate from and reproducible
    like the training seed (mirrors run_multiseed.run_quantgan_seed)."""
    set_all_seeds(GEN_SEED_OFFSET + seed)
    os.makedirs(RESULT_DIR, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(
            SET_SIZE, config.noise_dim, config.n_steps, device=config.device
        )
        fake = generator(noise)
        fake_data = inverse_scaling(fake, scalers).astype(np.float32)  # (2000, 6, 256)
    np.save(out_path, fake_data)
    print(f"[gen] seed {seed}: shape={fake_data.shape} -> {out_path}")


def ask_plan(args):
    """Resolve do_train from CLI flags, else interactive prompt. n_sets is fixed:
    the 5-seed scheme always makes one batch per model (5 models -> 5 batches)."""
    if args.train:
        return True
    if args.no_train:
        return False
    return input(
        "새로 훈련하시겠습니까? 기존 weight으로 생성만 하려면 n [y/N]: "
    ).strip().lower() in ("y", "yes")


def run_seed(seed, do_train, epochs, device):
    """Train (or load) one seed's QuantGAN and generate its single batch."""
    set_all_seeds(seed)
    config = make_config(epochs, device, seed=seed)
    train_dl, training_data, scalers = load_data(config)
    weight_path = os.path.join(RESULT_DIR, f"generator_best_seed{seed}.pt")
    ckpt_dir = os.path.join(WEIGHT_DIR, f"seed{seed}")  # transient per-seed checkpoints

    if do_train:
        generator = UserGenerator(config)
        discriminator = UserDiscriminator(config)
        trainer = GANTrainer(
            G=generator, D=discriminator, train_dl=train_dl, config=config
        )
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        trainer.results_dir = ckpt_dir
        shutil.rmtree(
            os.path.join(HERE, "results"), ignore_errors=True
        )  # trainer __init__ mkdir
        print(f"[run] seed {seed}: training QuantGAN for {config.n_epochs} epochs...")
        trainer.fit()  # ckpt_dir/Generator_{e}.pt
        generator = select_best_epoch(
            config, scalers, training_data, ckpt_dir
        )  # best epoch by EMD
        os.makedirs(RESULT_DIR, exist_ok=True)
        torch.save(generator.state_dict(), weight_path)
        print(f"[run] seed {seed}: generator weight saved: {weight_path}")
        shutil.rmtree(ckpt_dir, ignore_errors=True)  # drop intermediate checkpoints
    else:
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"기존 weight 없음: {weight_path} — 먼저 훈련(--train)하세요."
            )
        generator = UserGenerator(config).to(config.device)
        generator.load_state_dict(torch.load(weight_path, map_location=config.device))
        print(f"[run] seed {seed}: 기존 weight 로드: {weight_path}")

    generate_one(
        generator, scalers, config, seed, os.path.join(RESULT_DIR, f"{seed}.npy")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument(
        "--train", action="store_true", help="새로 훈련 (미지정 시 대화형 질문)"
    )
    ap.add_argument(
        "--no-train", action="store_true", help="훈련 없이 기존 weight으로 생성만"
    )
    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="훈련/생성할 시드 (기본 0..4)",
    )
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested but CUDA is not available.")

    do_train = ask_plan(args)
    print(f"[run] 5-seed scheme: seeds={args.seeds} -> one 2000 batch per model")
    for seed in args.seeds:
        print(f"\n{'='*60}\nQuantGAN seed={seed}\n{'='*60}")
        run_seed(seed, do_train, args.epochs, args.device)


if __name__ == "__main__":
    main()
