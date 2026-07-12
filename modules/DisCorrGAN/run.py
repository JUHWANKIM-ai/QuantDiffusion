"""
DisCorrGAN benchmark runner (unified convention).

Running `python run.py` end-to-end:
  1. reads the shared dataset  data/indices.csv  (repo root),
  2. trains DisCorrGAN — one TCN(+self-attention) generator/discriminator PER asset,
     WGAN-GP adversarial loss + pairwise correlation loss (original trainer logic),
  3. picks, PER asset, the checkpoint with the lowest train-set 100-day cumulative
     EMD as that asset's final generator (train-min-EMD, DisCorrGAN's native scheme),
  4. saves the best per-asset generators + 5 synthetic sets of 2000 samples to
        result/statistics/synthetic_data/discorrgan/
            0.npy ... 4.npy               # each (2000, 6, 256) float32
            generator_best_var_{i}.pt     # best-epoch generator per asset

This matches the diffusionTS / QuantGAN / TailGAN runners so all benchmarks share
the same output layout and are auto-discovered by 3_benchmark.ipynb.

    python run.py                 # full pipeline
    python run.py --epochs 60     # short test run

Notes
-----
* torch 1.9 has no torch.corrcoef (added in 1.10); the correlation loss needs it,
  so a numpy-equivalent shim is installed before training.
* wandb is disabled (WANDB_MODE=disabled) so training runs non-interactively.
"""

import argparse
import glob
import os
import re
import sys
from os.path import join

# --- disable wandb BEFORE the trainer (which imports wandb) is imported ---
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
os.chdir(HERE)                       # so `from src...` and ./configs/config.yaml resolve
sys.path.insert(0, HERE)

# torch.corrcoef shim (torch < 1.10) — matches numpy.corrcoef for a (F, N) input.
if not hasattr(torch, "corrcoef"):
    def _corrcoef(x):
        x = x - x.mean(dim=-1, keepdim=True)
        cov = x @ x.transpose(-1, -2)
        d = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1))
        cov = cov / d.unsqueeze(-1)
        cov = cov / d.unsqueeze(-2)
        return cov.clamp(-1.0, 1.0)
    torch.corrcoef = _corrcoef

from src.utils import scaling, inverse_scaling, rolling_window, set_seed, load_config
from src.baselines.networks.generators import TCNGenerator
from src.baselines.networks.discriminators import TCNDiscriminator
from src.baselines.trainer import GANTrainer, linear_rampup

DATA_CSV = join(PROJECT, "data", "indices.csv")           # shared dataset (repo root)
CONFIG_YAML = join(HERE, "configs", "config.yaml")
RESULT_DIR = join(PROJECT, "result", "statistics", "synthetic_data", "discorrgan")
WEIGHT_DIR = join(PROJECT, "result", "weight", "discorrgan")   # transient per-epoch checkpoints
TRAIN_RATIO = 0.7
N_SETS = 5            # number of synthetic sets (0.npy ... 4.npy)
SET_SIZE = 2000       # samples per set


def compute_avg_emd(real_bct, fake_bct, window=100):
    """EMD of `window`-day cumulative returns, summed over channels of the given arrays."""
    emd = 0.0
    for i in range(real_bct.shape[1]):
        real_dist = rolling_window(real_bct[:, i, :].T, window).sum(axis=1).ravel()
        fake_dist = rolling_window(fake_bct[:, i, :].T, window).sum(axis=1).ravel()
        emd += wasserstein_distance(real_dist, fake_dist)
    return emd


def load_data(config):
    """data/indices.csv -> train-split scaled windows + scalers (fit on train only)."""
    df = pd.read_csv(DATA_CSV)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df = df.set_index("Date").apply(pd.to_numeric).astype(float)

    log_returns = np.log1p(df.pct_change()).dropna().values     # (T, 6)
    train_raw = log_returns[:int(TRAIN_RATIO * len(log_returns))]   # time-ordered split

    train_scaled, scalers = scaling(train_raw)                  # gaussianized, fit on train
    n_steps = config.n_steps
    train_windowed = torch.from_numpy(rolling_window(train_scaled, n_steps)).float()   # (B, T, 6)
    train_dl = DataLoader(TensorDataset(train_windowed), batch_size=config.batch_size, shuffle=True)

    # unscaled train windows as (N, 6, T) — EMD reference (train, no look-ahead)
    train_org = torch.from_numpy(rolling_window(train_raw, n_steps)).float()
    real_bct = train_org.transpose(1, 2).cpu().numpy().astype(np.float32)
    print(f"[data] train windows: {tuple(train_windowed.shape)}  real_bct: {real_bct.shape}")
    return train_dl, real_bct, scalers


def build_models(config):
    G = {i: TCNGenerator(config).to(config.device) for i in range(config.n_vars)}
    D = {i: TCNDiscriminator(config).to(config.device) for i in range(config.n_vars)}
    return G, D


def train(config, train_dl, epochs, save_every, save_from):
    """Own training loop (mirrors GANTrainer.fit) with configurable checkpoint saving."""
    G, D = build_models(config)
    trainer = GANTrainer(G=G, D=D, train_dl=train_dl, config=config)
    import shutil
    shutil.rmtree(WEIGHT_DIR, ignore_errors=True)     # drop stale checkpoints before a fresh run
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    n_saved = 0
    from tqdm import tqdm
    for epoch in tqdm(range(epochs), desc="[discorrgan] train"):
        corr_weight = linear_rampup(epoch, config.rampup_epochs, config.corr_weight)
        for n, (real) in enumerate(train_dl):
            real_batch = real[0].transpose(1, 2).to(config.device)
            trainer.step(real_batch, n, corr_weight)
        if epoch % save_every == 0 and epoch >= save_from:
            for i in range(config.n_vars):
                torch.save(G[i].state_dict(), join(WEIGHT_DIR, f"Generator_{epoch}_var_{i}.pt"))
            n_saved += 1
    print(f"[train] saved {n_saved} checkpoint epochs -> {WEIGHT_DIR}")
    return G


def select_best_per_var(config, real_bct, scalers, sample_n):
    """For each asset i, pick the checkpoint epoch with lowest train EMD (DisCorrGAN native)."""
    def _ep(p):
        return int(re.search(r"_(\d+)_var_", os.path.basename(p)).group(1))

    device = config.device
    noise = torch.randn(sample_n, config.noise_dim, config.n_steps, device=device)
    best_paths, best_emds = [], []
    for i in range(config.n_vars):
        ckpts = sorted(glob.glob(join(WEIGHT_DIR, f"Generator_*_var_{i}.pt")), key=_ep)
        if not ckpts:
            raise RuntimeError(f"no checkpoints for var {i} in {WEIGHT_DIR}")
        gen = TCNGenerator(config).to(device)
        best_p, best_e = ckpts[-1], float("inf")
        for ck in ckpts:
            gen.load_state_dict(torch.load(ck, map_location=device))
            gen.eval()
            with torch.no_grad():
                fake = gen(noise)                                  # (sample_n, 1, T)
            fake_i = inverse_scaling(fake, scalers, i)             # (sample_n, 1, T) np
            emd = compute_avg_emd(real_bct[:sample_n, i:i + 1, :], fake_i, 100)
            if emd < best_e:
                best_e, best_p = emd, ck
        best_paths.append(best_p)
        best_emds.append(best_e)
        print(f"[select] var {i}: best {os.path.basename(best_p)}  EMD={best_e:.5f}")

    # persist best generators to the result dir
    os.makedirs(RESULT_DIR, exist_ok=True)
    for i, p in enumerate(best_paths):
        torch.save(torch.load(p, map_location="cpu"), join(RESULT_DIR, f"generator_best_var_{i}.pt"))
    print(f"[select] per-var mean EMD={np.mean(best_emds):.5f}  -> saved best generators")
    return best_paths


def _load_best_generators(config):
    gens = []
    for i in range(config.n_vars):
        p = join(RESULT_DIR, f"generator_best_var_{i}.pt")
        if not os.path.exists(p):
            raise FileNotFoundError(f"기존 weight 없음: {p} — 먼저 훈련(--train)하세요.")
        g = TCNGenerator(config).to(config.device)
        g.load_state_dict(torch.load(p, map_location=config.device))
        g.eval()
        gens.append(g)
    return gens


def generate_sets(config, scalers, n_sets):
    """Generate n_sets synthetic sets of SET_SIZE -> discorrgan/{0..n_sets-1}.npy."""
    gens = _load_best_generators(config)
    os.makedirs(RESULT_DIR, exist_ok=True)
    with torch.no_grad():
        for s in range(n_sets):
            noise = torch.randn(SET_SIZE, config.noise_dim, config.n_steps, device=config.device)
            per_var = [inverse_scaling(gens[i](noise), scalers, i) for i in range(config.n_vars)]
            fake_data = np.concatenate(per_var, axis=1).astype(np.float32)   # (SET_SIZE, 6, T)
            out_path = join(RESULT_DIR, f"{s}.npy")
            np.save(out_path, fake_data)
            print(f"[gen] set {s}: shape={fake_data.shape} -> {out_path}")


def ask_plan(args):
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
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--train", action="store_true", help="새로 훈련 (미지정 시 대화형 질문)")
    ap.add_argument("--no-train", action="store_true", help="훈련 없이 기존 weight으로 생성만")
    ap.add_argument("--sets", type=int, default=None, help="생성할 2000개 묶음 수 (미지정 시 질문)")
    ap.add_argument("--save-every", type=int, default=10, help="체크포인트 저장 주기(epoch)")
    ap.add_argument("--save-from", type=int, default=50, help="체크포인트 저장 시작 epoch")
    ap.add_argument("--select-samples", type=int, default=2000, help="best epoch EMD 선택 샘플 수")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested but CUDA is not available.")

    do_train, n_sets = ask_plan(args)

    config = load_config(CONFIG_YAML)
    config.device = args.device
    config.n_epochs = args.epochs
    set_seed(config.seed)
    print(f"[run] DisCorrGAN  n_vars={config.n_vars}, n_steps={config.n_steps}, "
          f"noise_dim={config.noise_dim}, hidden={config.hidden_dim}, epochs={args.epochs}")

    train_dl, real_bct, scalers = load_data(config)

    if do_train:
        save_from = min(args.save_from, max(args.epochs - args.save_every, 0))  # ensure something saves
        train(config, train_dl, args.epochs, args.save_every, save_from)
        select_best_per_var(config, real_bct, scalers, args.select_samples)

    print(f"[run] 2000개 x {n_sets}묶음 생성...")
    generate_sets(config, scalers, n_sets)


if __name__ == "__main__":
    main()
