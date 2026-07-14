# QuantDiffusion

**Multivariate Financial Scenario Generation for Portfolio Risk Assessment via Realized Volatility Decomposition**

Juhwan Kim, Soobeen Kim, JaeWook Song (Hanyang University) · *7th ACM International Conference on AI in Finance (ICAIF '26)*

QuantDiffusion is a denoising diffusion model for generating realistic multivariate financial return
scenarios. It is designed for risk-sensitive tasks (stress testing, deep hedging, Value-at-Risk / Expected
Shortfall estimation) where reproducing the **tail** of the return distribution matters as much as the bulk.

---

## Motivation

Two families dominate financial time-series generation:

- **GANs** reproduce stylized facts and cross-asset dependence well, but admit **no stochastic-process (SDE)
  interpretation** of the paths they produce.
- **Diffusion models** are intrinsically formulated as SDEs, yet existing ones fail to reproduce return
  stylized facts and, because training minimizes *average* error, struggle to model **ordinary and tail events
  simultaneously**.

Reading diffusion training as **Langevin dynamics**, the DDPM forward noise (Brownian) becomes statistically
indistinguishable from the Brownian footprint of a stochastic-volatility process at late noising stages, so the
score network wastes capacity separating signal from noise. QuantDiffusion removes this conflation by
**extracting the volatility component before training**.

## Method

Each return is factored into a standardized shock and a volatility magnitude, and the model only ever sees the
components:

```
r_t = z_t * sqrt(RV_t),      z_t = r_t / sqrt(RV_t)
```

where `RV_t` is the realized variance from a rolling **10-day close-to-close** window. This expands an
`n`-asset series into a `2n`-dimensional representation `(z, RV)`.

1. **Realized Volatility (RV) decomposition.** Via Itô's lemma the coupled `(r, v)` SDE decouples into a shock
   channel whose diffusion depends only on `z` and a log-volatility channel, flattening the energy landscape and
   stabilizing the reverse-time Langevin dynamics. Statistically, `r = z·sqrt(RV)` mirrors the definition of a
   Student-*t* variate (`Z / sqrt(V/ν)`), so the construction realizes a **dynamic Student-*t*** whose scale
   process is learned from data.
2. **Lambert *W* Gaussianization.** The residual `z` is Gaussianized with the inverse Lambert *W* transform.
   Because the map is `C²` and invertible, Itô's lemma applies and the transform acts at the SDE level
   (suppressing the jump term), rather than as a preprocessing heuristic.
3. **DMS-TCN backbone** (`models/DMS_TCN.py`, class `DMSTCN`). A Dilated Multi-Scale TCN with a large-kernel
   branch (dilations `[1, 4, 16]`) and a small-kernel branch, fused to capture local and long-horizon
   dependence jointly.
4. **Noise structure.** `ℓ1` loss, a **Student-*t* prior** noise distribution for tail fidelity, and a
   cosine schedule selected by the ANT score (`τ = 1.5`, `β ∈ [1e-5, 0.5]`, `T = 75`, `45` epochs).

Isolating the extreme mass of returns is what empirically lets ordinary and tail events be learned together; it
also points to a jump component, so the learned dynamics stay coherent with a **stochastic volatility jump
diffusion (SVJD)** reading, a property absent in GAN-based generators.

## Results

Real daily returns for six assets (DJIA, Nasdaq, JP Morgan, Hang Seng, Gold, WTI crude), evaluated against
**TailGAN, QuantGAN, and Diffusion-TS**. QuantDiffusion is best on 12 of 17 statistical-fidelity metrics and
top-2 on 15, and it markedly reduces portfolio-level tail-risk error.

**Marginal / temporal fidelity (lower is better; QD = QuantDiffusion):**

| Metric | TailGAN | QuantGAN | Diffusion-TS | **QD** |
|---|---|---|---|---|
| Kurt(1)   | 4.288 | 5.151 | 2.934 | **0.794** |
| ACF(\|x\|) | 0.0051 | 0.0038 | 0.0041 | **0.0015** |
| ACF(x²)   | 0.0030 | 0.0023 | 0.0023 | **0.0009** |
| Leverage  | 0.0016 | 0.0011 | 0.0012 | **0.0007** |

**Portfolio risk, relative error over 4 strategies (lower is better):**

| Metric | TailGAN | QuantGAN | Diffusion-TS | **QD** |
|---|---|---|---|---|
| RE VaR(0.01) | 0.561 | 0.675 | 0.411 | **0.335** |
| RE ES(0.01)  | 0.657 | 0.862 | 0.362 | **0.339** |

An ablation confirms the RV decomposition is the dominant driver of tail-risk fidelity, with the Lambert *W*
transform a secondary but consistent contributor.

---

## Repository layout

```
quantDiffusion/
├── 1_ablation_study.ipynb   # QuantDiffusion (baseline) + 3 ablations: train / generate / evaluate
├── 2_benchmark.ipynb        # QuantDiffusion vs QuantGAN / TailGAN / Diffusion-TS
├── 3_visualization.ipynb    # ACF / correlation / histogram figures + CRSP-vs-yfinance data check
├── ANT_score.ipynb          # ANT noise-schedule scoring (schedule selection)
├── models/
│   ├── DMS_TCN.py           # DMS-TCN backbone (class DMSTCN)
│   └── quantTCN.py          # plain TCN (used by the "w/o DMS-TCN" ablation)
├── modules/
│   ├── diffusion.py         # TimeSeriesGaussianDiffusion (DDPM, Student-t prior)
│   ├── QuantGAN/  TailGAN/  diffusionTS/   # baselines (each: run.py trains + generates)
│   └── TailGAN/tailgan_eval.py             # portfolio VaR/ES metric (paper's risk evaluation)
├── utils/
│   ├── file_manager.py  metric.py  utils.py  gaussianize.py  ANT_utils.py
│   └── eval_portfolio/     # legacy strategy eval (not part of the reported metrics)
├── data/
│   ├── indices.csv         # 6-asset CRSP close panel, 2001-01-01 .. 2025-12-31
│   ├── RV/                 # realized-volatility estimators (Close_to_Close, Parkinson, ...)
│   └── TailGAN/            # windows + thresholds + transition matrix for the TailGAN-style
│                           #   VaR/ES metric (git-ignored; regenerate via modules/TailGAN/run.py --train)
└── result/
    ├── statistics/synthetic_data/<tag>/{0..4}.npy   # generated scenarios per model
    ├── statistics/*.csv                             # summary tables (naming below)
    └── images/{acf,correlation,histogram}/          # figures
```

## Setup

Two conda environments are used (PyTorch CUDA stacks differ across baselines):

| Environment | Stack | Models |
|---|---|---|
| `research` | PyTorch 1.9.0 + cu111 | QuantDiffusion, QuantGAN, TailGAN |
| `abacus`   | PyTorch 2.x + `ema_pytorch` | Diffusion-TS |

Data (`data/indices.csv`, `data/RV/`) is CRSP daily prices for the six assets. The `data/TailGAN/` artifacts
required by the portfolio VaR/ES metric are regenerable with `python modules/TailGAN/run.py --train`.

## Usage

Run the notebooks with the `research` kernel (top to bottom). Notebooks **1** and **2** follow a **3-tier
generation order** per model, so nothing is recomputed unnecessarily:

1. if `result/statistics/synthetic_data/<tag>/` already has `.npy` → **evaluate only**;
2. else if a trained weight (`quantDiffusion_best*.pt`) exists → **load it and regenerate**;
3. else → **train from scratch, then generate**.

- **`1_ablation_study.ipynb`** — trains/generates QuantDiffusion (`baseline`) and the three ablations
  (`wo_rv`, `wo_lambertw`, `wo_moderntcn`) and writes their fidelity + TailGAN-style VaR/ES tables.
- **`2_benchmark.ipynb`** — evaluates QuantDiffusion against the baselines. Only QuantDiffusion can be
  regenerated/trained here; the external baselines (QuantGAN, TailGAN, Diffusion-TS) are evaluated from their
  existing `.npy` (produced by each `modules/<model>/run.py`).
- **`3_visualization.ipynb`** — saves ACF, correlation-heatmap, and cumulative-return-histogram figures, and
  cross-checks the CRSP dataset against yfinance.
- **`ANT_score.ipynb`** — scores noise schedules (used to select the cosine schedule above).

### Output naming

Summary CSVs under `result/statistics/` are grouped by prefix:

| Group | File(s) |
|---|---|
| Per-model (individual) | `individual__<tag>__gen_quality.csv` |
| Ablation summary | `ablation__gen_quality.csv`, `ablation__tailgan_vares.csv` |
| Benchmark summary | `benchmark__gen_quality.csv`, `benchmark__tailgan_vares.csv` |

## Code ↔ paper mapping

| Paper | Code |
|---|---|
| QuantDiffusion (ours) | tag `baseline`, config RV = Close-to-Close, window = 10 |
| RV decomposition `r = z·√RV` | `fit_transform_garch_12d` (in `1_ablation_study.ipynb`) |
| Lambert *W* Gaussianization | `utils/gaussianize.py`, `scaling` / `inverse_scaling` in `utils/utils.py` |
| DMS-TCN backbone | `models/DMS_TCN.py` (`DMSTCN`) |
| Diffusion (DDPM, Student-*t* prior) | `modules/diffusion.py` |
| Ablations: w/o RV Decomp. / w/o LambertW / w/o DMS-TCN | tags `wo_rv` / `wo_lambertw` / `wo_moderntcn` |
| Portfolio VaR / ES (relative error) | `modules/TailGAN/tailgan_eval.py` |
| Statistical fidelity metrics | `utils/metric.py` |
| ANT noise schedule | `ANT_score.ipynb`, `utils/ANT_utils.py` |
| Baselines | `modules/QuantGAN`, `modules/TailGAN`, `modules/diffusionTS` |

## Citation

```bibtex
@inproceedings{kim2026quantdiffusion,
  title     = {QuantDiffusion: Multivariate Financial Scenario Generation for
               Portfolio Risk Assessment via Realized Volatility Decomposition},
  author    = {Kim, Juhwan and Kim, Soobeen and Song, JaeWook},
  booktitle = {Proceedings of the 7th ACM International Conference on AI in Finance (ICAIF '26)},
  year      = {2026},
}
```
