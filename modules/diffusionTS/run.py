"""
Diffusion-TS benchmark runner (unified convention).

Running `python run.py` end-to-end:
  1. reads the shared dataset  data/indices.csv  (repo root),
  2. trains Diffusion_TS, checkpointing every `save_cycle` steps,
  3. picks the best checkpoint by 100-day cumulative EMD (no look-ahead: vs train data),
     mirroring the QuantGAN runner,
  4. trains once per seed (0..4) and saves, per seed, the best weight + ONE synthetic
     set of 2000 samples (1 model -> 1 batch) to
        result/statistics/synthetic_data/diffusionTS/
            0.npy ... 4.npy                 # one batch per model, each (2000, 6, 256) float32
            diffusion_ts_best_seed{s}.pt    # best-EMD checkpoint of seed s
  5. deletes the intermediate checkpoint folder after each seed.

Intermediate per-step checkpoints are written under  result/weight/  (not the
module dir) and pruned once the best is selected.

This matches the TailGAN / QuantGAN runners so all three benchmarks share
the same output layout.

Modes
-----
    python run.py                  # full pipeline, 5 seeds (train -> select -> 1 batch each)
    python run.py --train          # force (re)train all seeds
    python run.py --no-train       # 1 batch per seed from existing per-seed weights
    python run.py --prep-only      # only build the train/test .npy cache
"""

import argparse
import gc
import random
import os
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.io_utils import instantiate_from_config, load_yaml_config
from engine.solver import Trainer

DEFAULT_CONFIG = os.path.join(HERE, "Config", "indices2.yaml")
BENCHMARK = "diffusionTS"
RESULT_DIR = os.path.join(PROJECT, "result", "statistics", "synthetic_data", BENCHMARK)
WEIGHT_DIR = os.path.join(PROJECT, "result", "weight", BENCHMARK)  # transient checkpoints
SEEDS = [0, 1, 2, 3, 4]   # 5 models, one batch each -> 0.npy ... 4.npy
GEN_SEED_OFFSET = 1000    # generation reseeds at GEN_SEED_OFFSET + seed
SET_SIZE = 2000       # samples per set (one set per seed)


def set_all_seeds(seed):
    """Pin python / numpy / torch / cuda RNGs so each seed is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------- phase 1: data prep ----------
def rolling_window(data: np.ndarray, window_size: int) -> np.ndarray:
    n_windows = data.shape[0] - window_size + 1
    out = np.zeros((n_windows, window_size, data.shape[1]), dtype=np.float32)
    for idx in range(n_windows):
        out[idx] = data[idx : idx + window_size]
    return out


def pick_indices(n_available: int, n_target: int) -> np.ndarray:
    if n_available <= 0:
        return np.array([], dtype=int)
    if n_available <= n_target:
        return np.arange(n_available)
    return np.linspace(0, n_available - 1, n_target).round().astype(int)


def prepare_npy(dl_cfg: dict):
    csv_path = os.path.normpath(os.path.join(HERE, dl_cfg["data_root"]))
    npy_dir = os.path.normpath(os.path.join(HERE, dl_cfg["npy_dir"]))
    seq_len = int(dl_cfg["seq_length"])
    target_n = int(dl_cfg["target_n"])
    train_ratio = float(dl_cfg["train_ratio"])

    print(f"[prep] Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df.set_index("Date", inplace=True)
    df = df.apply(pd.to_numeric).astype(float)

    df_log_returns = np.log1p(df.pct_change()).dropna()
    log_returns = df_log_returns.values.astype(np.float32)
    print(f"[prep] Log-returns shape: {log_returns.shape}")

    train_size = int(train_ratio * len(log_returns))
    train_raw = log_returns[:train_size]
    test_raw = log_returns[train_size:]

    train_win = rolling_window(train_raw, seq_len)
    test_win = rolling_window(test_raw, seq_len)
    print(f"[prep] Train windows (B,T,C): {train_win.shape}")
    print(f"[prep] Test  windows (B,T,C): {test_win.shape}")

    train_sel = train_win[pick_indices(train_win.shape[0], target_n)]
    test_sel = test_win[pick_indices(test_win.shape[0], target_n)]

    train_bct = np.transpose(train_sel, (0, 2, 1)).astype(np.float32)
    test_bct = np.transpose(test_sel, (0, 2, 1)).astype(np.float32)

    os.makedirs(npy_dir, exist_ok=True)
    train_path = os.path.join(npy_dir, dl_cfg["train_name"])
    test_path = os.path.join(npy_dir, dl_cfg["test_name"])
    np.save(train_path, train_bct)
    np.save(test_path, test_bct)
    print(f"[prep] Saved {train_path}  shape={train_bct.shape}")
    print(f"[prep] Saved {test_path}  shape={test_bct.shape}")
    return train_path, test_path


# ---------- phase 2+3: training, selection & sampling ----------
class NpyWindowDataset(Dataset):
    """Loads a pre-built (B, C, T) .npy and serves (T, C) batches normalized to [-1, 1]."""

    def __init__(self, npy_path: str, auto_norm: bool = True):
        arr_bct = np.load(npy_path).astype(np.float32)  # (B, C, T)
        arr_btc = np.transpose(arr_bct, (0, 2, 1))      # (B, T, C)
        self.B, self.T, self.C = arr_btc.shape
        self.window = self.T
        self.var_num = self.C
        self.auto_norm = auto_norm

        flat = arr_btc.reshape(-1, self.C)
        self.scaler = MinMaxScaler().fit(flat)
        norm01 = self.scaler.transform(flat)
        norm = norm01 * 2.0 - 1.0 if auto_norm else norm01
        self.samples = norm.reshape(self.B, self.T, self.C).astype(np.float32)

    def __len__(self):
        return self.B

    def __getitem__(self, idx):
        return torch.from_numpy(self.samples[idx])


def all_milestones(folder: str):
    if not os.path.isdir(folder):
        return []
    ms = []
    for fn in os.listdir(folder):
        if fn.startswith("checkpoint-") and fn.endswith(".pt"):
            try:
                ms.append(int(fn[len("checkpoint-") : -len(".pt")]))
            except ValueError:
                pass
    return sorted(ms)


def generate_bct(ema_model, dataset, n, device):
    """Generate n samples and return them in original scale as (n, C, T)."""
    with torch.inference_mode():
        out = ema_model.generate_mts(batch_size=n)
    samples = out.detach().cpu().numpy()  # (n, T, C) in [-1, 1] if auto_norm
    del out
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if dataset.auto_norm:
        samples = unnormalize_to_zero_to_one(samples)
    flat = dataset.scaler.inverse_transform(samples.reshape(-1, dataset.C))
    samples_orig = flat.reshape(samples.shape).astype(np.float32)
    return np.transpose(samples_orig, (0, 2, 1)).astype(np.float32)  # (n, C, T)


def compute_avg_emd(real_bct, fake_bct, window=100):
    """Earth Mover's Distance of `window`-day cumulative returns, summed over features."""
    emd = 0.0
    for i in range(real_bct.shape[1]):
        real_dist = rolling_window(real_bct[:, i, :].T, window).sum(axis=1).ravel()
        fake_dist = rolling_window(fake_bct[:, i, :].T, window).sum(axis=1).ravel()
        emd += wasserstein_distance(real_dist, fake_dist)
    return emd


def select_best_milestone(trainer, dataset, real_bct, milestones, n_select, device):
    """Sample from each checkpoint's EMA model, keep the one with lowest EMD vs train data."""
    best_ms, best_emd = milestones[-1], float("inf")
    print(f"[select] evaluating {len(milestones)} checkpoints by EMD "
          f"(n_select={n_select} samples each)...")
    for ms in milestones:
        trainer.load(ms)
        ema_model = trainer.ema.ema_model
        ema_model.eval()
        fake_bct = generate_bct(ema_model, dataset, n_select, device)
        emd = compute_avg_emd(real_bct, fake_bct, window=100)
        marker = ""
        if emd < best_emd:
            best_emd, best_ms = emd, ms
            marker = "  <- best"
        print(f"[select] checkpoint-{ms} (step={trainer.step})  EMD={emd:.4f}{marker}")
        del fake_bct
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"[select] best = checkpoint-{best_ms}  EMD={best_emd:.4f}")
    return best_ms


def run_seed(config, do_train, seed, args):
    """Train (or load) one seed's Diffusion-TS and generate its single 2000 batch.
    1 model -> 1 batch, matching the quantDiffusion / QuantGAN / TailGAN 5-seed scheme."""
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)
    set_all_seeds(seed)
    print(f"[run] seed {seed}: device={device}  torch={torch.__version__}")

    dl_cfg = config["dataloader"]
    npy_dir = os.path.normpath(os.path.join(HERE, dl_cfg["npy_dir"]))
    train_path = os.path.join(npy_dir, dl_cfg["train_name"])
    dataset = NpyWindowDataset(train_path, auto_norm=bool(dl_cfg.get("auto_norm", True)))
    real_bct = np.load(train_path).astype(np.float32)

    dataloader = DataLoader(
        dataset, batch_size=int(dl_cfg["batch_size"]),
        shuffle=bool(dl_cfg.get("shuffle", True)), num_workers=0,
        pin_memory=(device.type == "cuda"), drop_last=True,
    )
    model = instantiate_from_config(config["model"]).to(device)

    save_dir = os.path.join(HERE, "OUTPUT", BENCHMARK, f"seed{seed}")
    runner_args = SimpleNamespace(name=BENCHMARK, save_dir=save_dir)
    trainer = Trainer(config=config, args=runner_args, model=model,
                      dataloader={"dataloader": dataloader, "dataset": dataset}, logger=None)
    ckpt_folder = str(trainer.results_folder)
    os.makedirs(RESULT_DIR, exist_ok=True)
    weight_dst = os.path.join(RESULT_DIR, f"diffusion_ts_best_seed{seed}.pt")

    if do_train:
        print(f"[run] seed {seed}: training for {config['solver']['max_epochs']} steps "
              f"(save_cycle={config['solver']['save_cycle']})...")
        trainer.train()
        if not all_milestones(ckpt_folder):
            trainer.milestone = 1
            trainer.save(trainer.milestone)
        milestones = all_milestones(ckpt_folder)
        if not milestones:
            raise RuntimeError(f"No checkpoints under {ckpt_folder}; training produced none.")
        best_ms = (args.milestone if args.milestone >= 0
                   else select_best_milestone(trainer, dataset, real_bct, milestones,
                                               args.select_samples, device))
        trainer.load(best_ms)
        shutil.copyfile(os.path.join(ckpt_folder, f"checkpoint-{best_ms}.pt"), weight_dst)
        print(f"[run] seed {seed}: best weight (checkpoint-{best_ms}) saved: {weight_dst}")
    else:
        if not os.path.exists(weight_dst):
            raise FileNotFoundError(f"기존 weight 없음: {weight_dst} — 먼저 훈련(--train)하세요.")
        data = torch.load(weight_dst, map_location=device)
        trainer.model.load_state_dict(data["model"])
        trainer.ema.load_state_dict(data["ema"])
        print(f"[run] seed {seed}: 기존 weight 로드: {weight_dst}")

    # ---- generate ONE batch of SET_SIZE for this seed ----
    ema_model = trainer.ema.ema_model
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    set_all_seeds(GEN_SEED_OFFSET + seed)
    chunks = []
    remaining = SET_SIZE
    while remaining > 0:
        n = min(args.num_samples, remaining)
        chunks.append(generate_bct(ema_model, dataset, n, device))
        remaining -= n
    batch = np.concatenate(chunks, axis=0)[:SET_SIZE].astype(np.float32)
    out_path = os.path.join(RESULT_DIR, f"{seed}.npy")
    np.save(out_path, batch)
    print(f"[run] seed {seed}: batch {batch.shape} -> {out_path}")

    if do_train and not args.keep_checkpoints and ckpt_folder and os.path.isdir(ckpt_folder):
        shutil.rmtree(ckpt_folder, ignore_errors=True)
        print(f"[run] seed {seed}: removed intermediate checkpoints: {ckpt_folder}")



def ask_train(args):
    """Resolve do_train (5-seed scheme is fixed at one batch per model)."""
    if args.train:
        return True
    if args.no_train:
        return False
    return input("새로 훈련하시겠습니까? 기존 weight으로 생성만 하려면 n [y/N]: ").strip().lower() in ("y", "yes")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="path to YAML config")
    p.add_argument("--prep-only", action="store_true", help="only build train/test npy")
    p.add_argument("--train", action="store_true", help="새로 훈련 (미지정 시 대화형 질문)")
    p.add_argument("--no-train", action="store_true", help="훈련 없이 기존 weight으로 생성만")
    p.add_argument("--milestone", type=int, default=-1,
                   help="force a checkpoint id (-1 = pick best by EMD)")
    p.add_argument("--select-samples", type=int, default=200,
                   help="samples drawn per checkpoint during EMD selection")
    p.add_argument("--num-samples", type=int, default=100, help="samples per file (= sampling batch)")
    p.add_argument("--keep-checkpoints", action="store_true",
                   help="keep the intermediate checkpoint folder (default: delete it)")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="compute device")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    print(f"[cfg] loaded: {args.config}")

    # checkpoints live under repo result/weight (transient; pruned after EMD selection)
    config["solver"]["results_folder"] = WEIGHT_DIR
    print(f"[cfg] results_folder -> {config['solver']['results_folder']}")

    prepare_npy(config["dataloader"])
    if args.prep_only:
        return

    do_train = ask_train(args)
    print(f"[main] 5-seed scheme: seeds={SEEDS} -> one 2000 batch per model")
    for seed in SEEDS:
        print(f"\n{'='*60}\nDiffusion-TS seed={seed}\n{'='*60}")
        run_seed(config, do_train, seed, args)


if __name__ == "__main__":
    main()
