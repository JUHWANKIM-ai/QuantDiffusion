"""
TailGAN benchmark runner (unified 5-seed convention).

Running `python run.py` end-to-end:
  1. reads the shared dataset  data/indices.csv  (repo root) and builds the
     rolling-window CSVs TailGAN expects,
  2. trains --models independent models (default 5, distinct seeds), original
     TailGAN.py logic, saving generators every 100 epochs,
  3. keeps each model's FINAL epoch (the official TailGAN checkpoint; no
     gen-quality selection, no convergence screening, no ensembling),
  4. generates ONE synthetic set per model (model m -> batch m, no pooling) to
        result/statistics/synthetic_data/tailgan/
            0.npy ... 4.npy               # one batch per model, each (2000, 6, 256) float32
            generator_best_id{m}.pt       # final-epoch generator of model m

One model -> one batch (5 models -> 5 batches), matching the quantDiffusion /
QuantGAN / diffusionTS runners so every model follows the same 5-seed scheme.

    python run.py                 # full pipeline (5 models)
    python run.py --epochs 50     # short test run
    python run.py --no-train      # regenerate batches from existing per-model weights
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
N_MODELS = 5          # 5 models (seeds), one batch each -> 0.npy ... 4.npy
GEN_SEED_OFFSET = 1000  # generation reseeds np.random at GEN_SEED_OFFSET + model index
SET_SIZE = 2000       # samples per set (one set per model)
STATIC_PORT_SEED = 0  # see ensure_static_basket


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


def ensure_static_basket(gsp, n_rows=N_ROWS, n_ports=50, static_way="LShort"):
    """Put the evaluation basket in place, never redrawing it.

    Delegates to tailgan_eval so training and evaluation cannot drift onto
    different baskets; see tailgan_eval.ensure_static_basket for why redrawing it
    invalidates every number in the result tables.
    """
    import tailgan_eval

    gsp.your_path = YOUR_PATH
    gsp.trans_parent_data_path = join(YOUR_PATH, "Static_Port_Transform")
    os.makedirs(gsp.trans_parent_data_path, exist_ok=True)
    return tailgan_eval.ensure_static_basket(PROJECT, gsp, n_rows=n_rows,
                                             n_ports=n_ports,
                                             static_way=static_way)


def last_epoch_for_model(ckpt_dir, m):
    """The final (highest-epoch) checkpoint of model m -> official TailGAN weight, no selection."""
    def _ep(pth):
        return int(re.search(r"_E(\d+)", pth).group(1))
    ckpts = sorted(glob.glob(join(ckpt_dir, f"generator_id{m}_E*.pt")), key=_ep)
    if not ckpts:
        raise RuntimeError(f"model {m}: no generator checkpoints were saved during training")
    return ckpts[-1]


def generate_per_model(TG, weight_paths, set_size=SET_SIZE):
    """Non-pooled generation: model m's weight -> its own batch m (set_size, 6, 256).
    Reseeds np.random at GEN_SEED_OFFSET + m so every batch is reproducible, mirroring the
    quantDiffusion / QuantGAN 5-seed generation convention (1 model -> 1 batch)."""
    import torch
    os.makedirs(RESULT_DIR, exist_ok=True)
    for m, p in enumerate(sorted(weight_paths)):
        g = TG.Generator()
        if TG.cuda:
            g.cuda()
        g.load_state_dict(torch.load(p, map_location="cuda" if TG.cuda else "cpu"))
        g.eval()
        np.random.seed(GEN_SEED_OFFSET + m)
        with torch.no_grad():
            fake = g(_tail_noise(TG, set_size)).cpu().numpy().astype(np.float32)
        out_path = join(RESULT_DIR, f"{m}.npy")
        np.save(out_path, fake)
        print(f"[gen] model {m}: shape={fake.shape} (from {os.path.basename(p)}) -> {out_path}")


def train_single_warm(TG, o, dataloader, model_index, seed, warmup_steps, warmup_lr):
    """TailGAN.Train_Single plus a discriminator warmup and per component logging.

    Why this exists. The discriminator is the VaR/ES estimator: the generator's loss
    is S(D(fake), real PNL), so it only carries signal once D reads the tail off a
    sorted P&L vector correctly. Its own objective is real_score - fake_score, which
    it can lower by inflating fake_score instead of becoming accurate, and at
    lr_D <= 1e-5 that is what happens: D stays at its random init, and from a random
    D the generator's loss falls monotonically in the generated volatility, telling
    it to be more extreme without bound. Measured on our data: loss_G 8.43 at 0.2x
    volatility down to 8.35 at 2x, against a floor of 2.11.

    Warming D up on real data alone first turns that into a bowl whose minimum sits
    at the real volatility (measured: argmin at 0.85x). Upstream Train_Single is left
    untouched; this function mirrors it, saving generators and loss_id{m}.npy in the
    same layout so screen_ensemble and select_ensemble keep working.
    """
    import time
    import random

    import numpy as np
    import torch
    from torch.autograd import Variable

    # per-model reproducibility: pin all four RNGs (statistical, matching the notebook
    # and QuantGAN/Diffusion-TS runners; cuDNN is left nondeterministic as in those)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    Tensor = TG.Tensor
    generator, discriminator, criterion = TG.Generator(), TG.Discriminator(), TG.Score()
    if TG.cuda:
        generator.cuda()
        discriminator.cuda()

    def noise(n):
        if "t" in o.noise_name:
            df = int(o.noise_name.split("t")[1])
            return Variable(Tensor(np.random.standard_t(df, (n, o.latent_dim))))
        return Variable(Tensor(np.random.normal(0, 1, (n, o.latent_dim))))

    def emp_floor(PNL):
        """Score at the empirical (VaR, ES) of the real P&L: the best G can reach."""
        X = PNL.T.detach()
        total = 0.0
        for alpha in o.alphas:
            v = torch.quantile(X, alpha, dim=1, keepdim=True)
            e = torch.stack([x[x <= vt].mean() for x, vt in zip(X, v)]).reshape(-1, 1)
            total += TG.S_quant(v, e, X, alpha).item()
        return total

    # ---- phase 1: D alone on real data, so it starts as a working VaR/ES estimator
    start_time = time.time()
    warm_opt = torch.optim.Adam(discriminator.parameters(), lr=warmup_lr,
                                betas=(o.b1, o.b2))
    it = iter(dataloader)
    for step in range(warmup_steps):
        try:
            R = next(it)
        except StopIteration:
            it = iter(dataloader)
            R = next(it)
        PNL, validity = discriminator(Variable(R.type(Tensor)))
        loss = criterion(validity, PNL)
        warm_opt.zero_grad()
        loss.backward()
        warm_opt.step()
        if step % 100 == 0 or step == warmup_steps - 1:
            fl = emp_floor(PNL)
            print(f"[warmup {step:4d}] D score={loss.item():8.4f}  floor={fl:8.4f}  "
                  f"gap={loss.item() - fl:+8.4f}")

    # fresh Adam moments for the adversarial phase, D already accurate
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=o.lr_G, betas=(o.b1, o.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=o.lr_D, betas=(o.b1, o.b2))

    # ---- phase 2: adversarial, same updates as upstream Train_Single
    loss_d_l, loss_g_l, diag_l = [], [], []
    for epoch in range(o.n_epochs):
        ep_d, ep_g, ep_real, ep_floor = [], [], [], []

        for i, R in enumerate(dataloader):
            real_R = Variable(R.type(Tensor))
            gen_R = generator(noise(R.shape[0]))

            if i % o.n_critic_D == 0:
                optimizer_D.zero_grad()
                PNL, PNL_validity = discriminator(real_R)
                gen_PNL, gen_PNL_validity = discriminator(gen_R)
                real_score = criterion(PNL_validity, PNL)
                fake_score = criterion(gen_PNL_validity, PNL)
                loss_D = real_score - fake_score
                loss_D.backward(retain_graph=True)
                optimizer_D.step()
                ep_d.append(loss_D.item())
                ep_real.append(real_score.item())
                ep_floor.append(emp_floor(PNL))

            if i % o.n_critic_G == 0:
                optimizer_G.zero_grad()
                gen_PNL, gen_PNL_validity = discriminator(gen_R)
                loss_G = criterion(gen_PNL_validity, PNL)
                loss_G.backward()
                optimizer_G.step()
                ep_g.append(loss_G.item())

        loss_d_l.append(np.mean(ep_d))
        loss_g_l.append(np.mean(ep_g))
        diag_l.append([np.mean(ep_real), np.mean(ep_floor)])

        if epoch % 100 == 0 or epoch == o.n_epochs - 1:
            # real_score - floor is D's accuracy; loss_G - floor is how far G still is
            r, f = diag_l[-1]
            with torch.no_grad():
                smp = generator(noise(500))
            std = smp.std(dim=(0, 2)).cpu().numpy()
            print(f"[Epoch {epoch}] [D loss: {loss_d_l[-1]: .4f}] "
                  f"[G loss: {loss_g_l[-1]:.4f}] [floor: {f:.4f}] "
                  f"[D err: {r - f:+.4f}] [G gap: {loss_g_l[-1] - f:+.4f}] "
                  f"[gen std: {np.round(std, 4)}] "
                  f"--- {int(time.time() - start_time)}s ---")

        if epoch % 100 == 0 or epoch == o.n_epochs - 1:
            torch.save(discriminator.state_dict(),
                       join(TG.model_path, "discriminator_id%d_E%d" % (model_index, epoch)))
            torch.save(generator.state_dict(),
                       join(TG.model_path, "generator_id%d_E%d" % (model_index, epoch)))

    np.save(join(TG.gen_data_path, "loss_id%d.npy" % model_index),
            np.stack([np.array(loss_d_l), np.array(loss_g_l)]))
    np.save(join(TG.gen_data_path, "diag_id%d.npy" % model_index),
            np.array(diag_l).T)      # rows = [real_score, empirical floor]


def ask_train(args):
    """Resolve do_train from CLI flags, else interactive prompt. The 5-seed scheme is
    fixed at one batch per model (N_MODELS models -> N_MODELS batches), so there is no
    set count to ask for."""
    if args.train:
        return True
    if args.no_train:
        return False
    return input("새로 훈련하시겠습니까? 기존 weight으로 생성만 하려면 n [y/N]: ").strip().lower() in ("y", "yes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("TG_EPOCHS", "20000")))
    ap.add_argument("--batch", type=int, default=None, help="batch size (default: TailGAN default)")
    ap.add_argument("--lr-g", type=float, default=None,
                    help="generator lr (default: TailGAN's own 1e-6)")
    ap.add_argument("--lr-d", type=float, default=None,
                    help="discriminator lr (default: TailGAN's own 1e-7)")
    ap.add_argument("--d-warmup", type=int, default=400,
                    help="adversarial 학습 전 D 단독 warmup step 수 (0이면 원본 그대로)")
    ap.add_argument("--lr-d-warmup", type=float, default=1e-3,
                    help="D warmup 단계의 lr (measured: 1e-3에서 100 step 내 floor 도달)")
    ap.add_argument("--train", action="store_true", help="새로 훈련 (미지정 시 대화형 질문)")
    ap.add_argument("--no-train", action="store_true", help="훈련 없이 기존 weight으로 생성만")
    ap.add_argument("--models", type=int, default=N_MODELS,
                    help="모델(=시드) 개수 (기본 5; 모델당 배치 1개, pooling 없음)")
    args = ap.parse_args()

    do_train = ask_train(args)

    # TailGAN uses flat imports (from Dataset import ...) and relative paths -> run from its dir.
    sys.path.insert(0, TG_DIR)
    os.chdir(TG_DIR)
    sys.argv = ["TailGAN.py"]                       # let argparse keep its defaults
    os.makedirs(RESULT_DIR, exist_ok=True)

    if do_train:
        n_windows = build_windows()

        # ---- static-portfolio matrices + path injection (training only) ----
        import gen_static_port as gsp
        ensure_static_basket(gsp)

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
        o.numNN = args.models                       # 5 models, one batch each (no ensembling)
        o.n_epochs = args.epochs
        if args.batch is not None:
            o.batch_size = args.batch
        if args.lr_g is not None:
            o.lr_G = args.lr_g
        if args.lr_d is not None:
            o.lr_D = args.lr_d
        # DataLoader has no drop_last; trim len to a multiple of batch so the last
        # partial batch can't mismatch Discriminator(Linear(batch_size)).
        o.len = (n_windows // o.batch_size) * o.batch_size
        TG.R_shape = (o.n_rows, o.n_cols)
        TG.PNL_shape = TG.Infer_Shape(TG.R_shape)
        TG.gen_data_path = join(YOUR_PATH, "Gens", DATA_NAME)
        TG.model_path = join(YOUR_PATH, "Models", DATA_NAME)
        os.makedirs(TG.gen_data_path, exist_ok=True)
        os.makedirs(TG.model_path, exist_ok=True)
        print(f"[run] R_shape={TG.R_shape}, len={o.len}, n_epochs={o.n_epochs}, "
              f"batch={o.batch_size}, steps/epoch={o.len // o.batch_size}, "
              f"lr_G={o.lr_G}, lr_D={o.lr_D}, "
              f"d_warmup={args.d_warmup}@{args.lr_d_warmup:g}, models={o.numNN}")
        shutil.rmtree(join(TG_DIR, "your_path"), ignore_errors=True)  # clean stray import-time dir

        # ---- keep every per-100-epoch generator under result/weight (transient) ----
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
        # ---- train N models (distinct seeds), saving generators every 100 epochs ----
        from Dataset import Dataset_IS
        base_seed = getattr(TG, "seed", 1)
        dataset = Dataset_IS(tickers=o.tickers, data_path=join(YOUR_PATH, "gan_data", o.data_name), length=o.len)
        dataloader = _torch.utils.data.DataLoader(dataset, batch_size=o.batch_size, shuffle=True)
        # each model's FINAL-epoch weight is written straight to synthetic_data/tailgan the
        # moment that model finishes training, so a completed model is visible immediately
        # (no waiting for the whole run). Official TailGAN scheme: final epoch, no selection.
        os.makedirs(RESULT_DIR, exist_ok=True)
        for fp in glob.glob(join(RESULT_DIR, "generator_best*.pt")):
            os.remove(fp)
        for m in range(o.numNN):
            print(f"------ Model {m}/{o.numNN} (seed {base_seed + m}) ------")
            if args.d_warmup > 0:
                train_single_warm(TG, o, dataloader, model_index=m, seed=base_seed + m,
                                  warmup_steps=args.d_warmup, warmup_lr=args.lr_d_warmup)
            else:
                TG.Train_Single(o, dataloader, model_index=m, seed=base_seed + m)
            last = last_epoch_for_model(tmp_ckpt, m)
            shutil.copyfile(last, join(RESULT_DIR, f"generator_best_id{m}.pt"))
            print(f"[save] model {m}: {os.path.basename(last)} -> synthetic_data/tailgan/generator_best_id{m}.pt")
        _np.save, _torch.save = _orig_np_save, _orig_torch_save
        shutil.rmtree(tmp_ckpt, ignore_errors=True)
        weight_paths = sorted(glob.glob(join(RESULT_DIR, "generator_best_id*.pt")))
        print(f"[run] trained {len(weight_paths)} models (final-epoch each, saved incrementally)")
    else:
        weight_paths = sorted(glob.glob(join(RESULT_DIR, "generator_best_id*.pt")))
        if not weight_paths:
            raise FileNotFoundError(f"기존 weight 없음: {RESULT_DIR}/generator_best_id*.pt — 먼저 훈련(--train)하세요.")
        import TailGAN as TG                           # generation needs only the Generator class
        TG.your_path = YOUR_PATH
        TG.opt.n_rows, TG.opt.n_cols = N_ROWS, N_COLS
        TG.R_shape = (N_ROWS, N_COLS)
        shutil.rmtree(join(TG_DIR, "your_path"), ignore_errors=True)
        print(f"[run] 기존 모델 weight {len(weight_paths)}개 로드")

    print(f"[run] {len(weight_paths)}개 모델 -> 모델당 2000개 배치 1개 생성 (pooling 없음)...")
    generate_per_model(TG, weight_paths)



if __name__ == "__main__":
    main()
