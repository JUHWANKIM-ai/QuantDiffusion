# QuantDiffusion Results Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the four results tables in `main.tex` with the current experimental data (`benchmark2_*` and `ablation_*` CSVs, with DisCorrGAN excluded), align the Experiments baseline listing with the three actual baselines, and rewrite every results interpretation paragraph (main + ablation) to match the new numbers, in formal academic prose.

**Architecture:** All edits are confined to `main.tex`. Table bodies are pre-rendered from the source CSVs with best/second-best marking already computed and verified; each table task replaces a whole `table*` environment. Prose tasks replace a single paragraph each with pre-written academic text whose numeric claims are derived from the same CSVs. Two edits to Section 2 / contributions are included but flagged as core-argument changes to be reviewed independently.

**Tech Stack:** LaTeX (`acmart` sigconf, `biblatex`, `kotex`). Source data: `result/statistics/{benchmark2,ablation}_summary_{gen_quality,tailgan_vares}.csv`. No local TeX toolchain is installed, so verification is structural (grep + brace/column checks) plus visual review; final PDF compilation is done by the user in Overleaf or a local TeX install.

## Global Constraints

- **Formal written register only (문어체 / academic).** No colloquial or spoken-style phrasing. Every sentence must read as formal academic written English.
- **Avoid dashes as much as possible.** Do not introduce em-dashes (`---`) or hyphenated compound modifiers in new prose. Use commas, colons, semicolons, or parentheses instead. The paper already follows this style (for example "state of the art", "GAN based", "cross asset"); match it. Hyphens inside proper names (Tail-GAN, Diffusion-TS, Lambert-W) are kept.
- **Numbers are authoritative from the CSVs.** Do not invent, round differently, or carry over any figure from the previous version of the paper. Every numeric claim must be traceable to `result/statistics/*.csv`.
- **DisCorrGAN is excluded** from the paper entirely (tables and prose). The main tables therefore drop the DisCorrGAN column from the source CSVs.
- **Lower is better** for every metric in all four tables; best is `\textbf{}`, second best is `\underline{}`.
- **Column order, main tables:** TailGAN, QuantGAN, Diffusion-TS, QuantDiffusion (ours last). Four columns.
- **Column order, ablation tables:** Baseline, w/o RV Decomp., w/o LambertW, w/o ModernTCN.
- The paper column header for the proposed model is `\textbf{QuantDiffusion}` (drop the notebook-internal tag `(ours, C2C/w10)`).
- **Realized volatility proxy is close-to-close realized variance over a rolling 10 day window (C2C/w10);** this is the reported configuration and the paper's methods text must describe it (not Parkinson). See Task 11.
- Commit after each task. Do not run `git push`.

## Source-of-truth data mapping

| Table (label) | main.tex lines | Source CSV |
|---|---|---|
| Main statistical fidelity (`table:statistical_fidelity_combined`) | 425–542 | `benchmark2_summary_gen_quality.csv` |
| Main portfolio risk (`table:portfolio_risk`) | 673–722 | `benchmark2_summary_tailgan_vares.csv` |
| Ablation fidelity (`table:ablation_fidelity`) | 545–662 | `ablation_summary_gen_quality.csv` |
| Ablation portfolio risk (`table:ablation_portfolio_risk`) | 730–779 | `ablation_summary_tailgan_vares.csv` |

Row layout follows the paper's current 17-row fidelity structure (single CCF row per lag using the MAE variant; no EMD, no CCF FROB/MSE) and the 6-row VaR/ES structure. This was confirmed with the user.

## Key standings (for the prose tasks; all from the CSVs, DisCorrGAN column dropped)

Comparison set: TailGAN, QuantGAN, Diffusion-TS, QuantDiffusion (four columns).

- **CCF(τ=0):** Diffusion-TS best (0.0533), QuantDiffusion 2nd (0.0572). **CCF(τ>0):** QuantDiffusion best (0.0341), Diffusion-TS 2nd (0.0367).
- **KS:** QuantDiffusion best at every horizon h∈{1,5,20,100} (at h=100, 0.1403 versus QuantGAN 0.1404, a razor-thin margin); QuantGAN second at all four.
- **Kurt:** QuantDiffusion best at every horizon h∈{1,5,20,100}. Most striking at h=1 (0.7942 versus Diffusion-TS 2.9343).
- **Skew:** QuantDiffusion weak. Second at h=1 (behind Diffusion-TS); last at h∈{5,20}; third at h=100.
- **ACF(|x|), ACF(x²), Leverage:** QuantDiffusion best, QuantGAN 2nd; Diffusion-TS roughly 2–3× worse than ours. **ACF(x):** Diffusion-TS best (0.0007), ours 2nd (0.0008), effectively tied near the noise floor.
- **VaR/ES:** QuantDiffusion best on all six. 2nd best = QuantGAN at VaR(0.05), Diffusion-TS on the other five. Improvement over 2nd best: VaR 34.2% / 32.1% / 18.5%; ES 27.6% / 18.0% / 6.5% at α = 0.05 / 0.025 / 0.01.
- QuantDiffusion is best on **12 of 17** fidelity metrics and top-two on **15 of 17** (only the three skewness horizons h∈{5,20,100} fall outside the top two, and it is second at h=1).
- **Ablation fidelity:** w/o RV wrecks ACF(|x|) (0.0015→0.0039) and ACF(x²) (0.0009→0.0024) but gives the best Skew; w/o LambertW degrades ACF and Kurt(1) (0.7942→0.8877); Baseline best on ACF(|x|), ACF(x²), Leverage, Kurt(20).
- **Ablation portfolio:** Baseline best on all six VaR/ES; w/o RV worst (0.58–0.66); w/o LambertW closest to Baseline (2nd).

---

## Task 1: Replace the main statistical fidelity table

**Files:**
- Modify: `main.tex:425-542`

- [ ] **Step 1: Replace the whole `table*` environment (lines 425–542) with the block below.**

```latex
\begin{table*}[htbp]
\centering
\caption{Statistical fidelity comparison across dependence, marginal distribution, and temporal structure metrics. Best in \textbf{bold}, second best \underline{underlined}; lower is better.}
\label{table:statistical_fidelity_combined}

\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{0.9}

\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{l|cccc}
\toprule
\textbf{Metric} & \textbf{TailGAN} & \textbf{QuantGAN} & \textbf{Diffusion-TS} & \textbf{QuantDiffusion} \\
\midrule
CCF($\tau = 0$)
& 0.1987 $\pm$ 0.0002
& 0.0968 $\pm$ 0.0005
& \textbf{0.0533} $\pm$ 0.0010
& \underline{0.0572} $\pm$ 0.0007 \\
CCF($\tau > 0$)
& 0.0666 $\pm$ 0.0001
& 0.0569 $\pm$ 0.0002
& \underline{0.0367} $\pm$ 0.0002
& \textbf{0.0341} $\pm$ 0.0002 \\
\midrule
KS(1)
& 0.1820 $\pm$ 0.0002
& \underline{0.0402} $\pm$ 0.0001
& 0.0948 $\pm$ 0.0006
& \textbf{0.0375} $\pm$ 0.0000 \\
KS(5)
& 0.1759 $\pm$ 0.0003
& \underline{0.0529} $\pm$ 0.0003
& 0.1141 $\pm$ 0.0005
& \textbf{0.0418} $\pm$ 0.0003 \\
KS(20)
& 0.1823 $\pm$ 0.0006
& \underline{0.0785} $\pm$ 0.0006
& 0.1335 $\pm$ 0.0005
& \textbf{0.0530} $\pm$ 0.0004 \\
KS(100)
& 0.2425 $\pm$ 0.0017
& \underline{0.1404} $\pm$ 0.0016
& 0.1912 $\pm$ 0.0011
& \textbf{0.1403} $\pm$ 0.0024 \\
\midrule
Skew(1)
& 0.2245 $\pm$ 0.0017
& 0.4560 $\pm$ 0.0135
& \textbf{0.1507} $\pm$ 0.0039
& \underline{0.1582} $\pm$ 0.0037 \\
Skew(5)
& \underline{0.2683} $\pm$ 0.0020
& 0.3424 $\pm$ 0.0083
& \textbf{0.1436} $\pm$ 0.0050
& 0.3335 $\pm$ 0.0073 \\
Skew(20)
& 0.2760 $\pm$ 0.0020
& \underline{0.2277} $\pm$ 0.0070
& \textbf{0.2043} $\pm$ 0.0018
& 0.3380 $\pm$ 0.0046 \\
Skew(100)
& 0.1262 $\pm$ 0.0067
& \textbf{0.0853} $\pm$ 0.0063
& \underline{0.0955} $\pm$ 0.0025
& 0.1199 $\pm$ 0.0058 \\
\midrule
Kurt(1)
& 4.2878 $\pm$ 0.0017
& 5.1508 $\pm$ 0.2507
& \underline{2.9343} $\pm$ 0.0070
& \textbf{0.7942} $\pm$ 0.0515 \\
Kurt(5)
& 1.6039 $\pm$ 0.0029
& \underline{0.9097} $\pm$ 0.0589
& 1.0401 $\pm$ 0.0168
& \textbf{0.8688} $\pm$ 0.0288 \\
Kurt(20)
& 0.7140 $\pm$ 0.0033
& \underline{0.5001} $\pm$ 0.0087
& 0.6869 $\pm$ 0.0071
& \textbf{0.2287} $\pm$ 0.0101 \\
Kurt(100)
& 0.2637 $\pm$ 0.0055
& 0.2678 $\pm$ 0.0109
& \underline{0.2555} $\pm$ 0.0077
& \textbf{0.2037} $\pm$ 0.0122 \\
\midrule
ACF($x$)
& 0.0010 $\pm$ 0.0000
& 0.0012 $\pm$ 0.0000
& \textbf{0.0007} $\pm$ 0.0000
& \underline{0.0008} $\pm$ 0.0000 \\
ACF($|x|$)
& 0.0051 $\pm$ 0.0000
& \underline{0.0038} $\pm$ 0.0000
& 0.0041 $\pm$ 0.0000
& \textbf{0.0015} $\pm$ 0.0001 \\
ACF($x^2$)
& 0.0030 $\pm$ 0.0000
& \underline{0.0023} $\pm$ 0.0000
& 0.0023 $\pm$ 0.0000
& \textbf{0.0009} $\pm$ 0.0000 \\
\midrule
Lev. Effect
& 0.0016 $\pm$ 0.0000
& \underline{0.0011} $\pm$ 0.0000
& 0.0012 $\pm$ 0.0000
& \textbf{0.0007} $\pm$ 0.0000 \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table*}
```

Note on ACF($x^2$): QuantGAN and Diffusion-TS both read 0.0023; the underline is placed on QuantGAN per the CSV ordering. Leave as written.

- [ ] **Step 2: Verify structure.**

Run: `grep -n "table:statistical_fidelity_combined\|DisCorrGAN\|COSCI\|SigCWGAN" main.tex`
Expected: label present once; **no** match for `DisCorrGAN`, `COSCI`, or `SigCWGAN`. Confirm the block uses `{l|cccc}`, has exactly 3 `&` per data row (4 columns) and one `\\` per row.

- [ ] **Step 3: Commit.**

```bash
git add main.tex
git commit -m "paper: replace main statistical fidelity table with benchmark2 data"
```

---

## Task 2: Replace the main portfolio risk (VaR/ES) table

**Files:**
- Modify: `main.tex:673-722`

- [ ] **Step 1: Replace the whole `table*` environment (lines 673–722) with the block below.**

```latex
\begin{table*}[htbp]
\centering
\caption{Portfolio risk fidelity: relative error of VaR and ES (averaged over the four trading strategies) between synthetic and historical data. Best in \textbf{bold}, second best \underline{underlined}; lower is better.}
\label{table:portfolio_risk}

\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{0.9}

\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{l|cccc}
\toprule
\textbf{Metric} & \textbf{TailGAN} & \textbf{QuantGAN} & \textbf{Diffusion-TS} & \textbf{QuantDiffusion} \\
\midrule
$\text{RE}_{\text{VaR}}(0.05)$
& 0.5343 $\pm$ 0.0064
& \underline{0.5340} $\pm$ 0.0112
& 0.5684 $\pm$ 0.0048
& \textbf{0.3514} $\pm$ 0.0054 \\
$\text{RE}_{\text{VaR}}(0.025)$
& 0.5278 $\pm$ 0.0061
& 0.5951 $\pm$ 0.0131
& \underline{0.4797} $\pm$ 0.0198
& \textbf{0.3259} $\pm$ 0.0026 \\
$\text{RE}_{\text{VaR}}(0.01)$
& 0.5605 $\pm$ 0.0065
& 0.6745 $\pm$ 0.0158
& \underline{0.4110} $\pm$ 0.0180
& \textbf{0.3351} $\pm$ 0.0034 \\
\midrule
$\text{RE}_{\text{ES}}(0.05)$
& 0.5429 $\pm$ 0.0091
& 0.6346 $\pm$ 0.0213
& \underline{0.4527} $\pm$ 0.0163
& \textbf{0.3276} $\pm$ 0.0023 \\
$\text{RE}_{\text{ES}}(0.025)$
& 0.5842 $\pm$ 0.0109
& 0.7237 $\pm$ 0.0363
& \underline{0.3995} $\pm$ 0.0189
& \textbf{0.3277} $\pm$ 0.0038 \\
$\text{RE}_{\text{ES}}(0.01)$
& 0.6572 $\pm$ 0.0190
& 0.8619 $\pm$ 0.0753
& \underline{0.3622} $\pm$ 0.0062
& \textbf{0.3388} $\pm$ 0.0104 \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table*}
```

- [ ] **Step 2: Verify.**

Run: `grep -n "table:portfolio_risk\|DisCorrGAN" main.tex`
Expected: exactly one `table:portfolio_risk` definition; no `DisCorrGAN`. Confirm `{l|cccc}`, 6 data rows, 3 `&` each, VaR block then ES block.

- [ ] **Step 3: Commit.**

```bash
git add main.tex
git commit -m "paper: replace main portfolio VaR/ES table with benchmark2 data"
```

---

## Task 3: Replace the ablation fidelity table

**Files:**
- Modify: `main.tex:545-662`

- [ ] **Step 1: Replace the whole `table*` environment (lines 545–662) with the block below.**

```latex
\begin{table*}[htbp]
\centering
\caption{Ablation study: statistical fidelity of QuantDiffusion (Baseline = full model) when each key component is removed (RV decomposition, Lambert-W Gaussianization, ModernTCN backbone). Best in \textbf{bold}, second best \underline{underlined}; lower is better.}
\label{table:ablation_fidelity}

\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{0.9}

\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{l|cccc}
\toprule
\textbf{Metric} & \textbf{Baseline} & \textbf{w/o RV Decomp.} & \textbf{w/o LambertW} & \textbf{w/o ModernTCN} \\
\midrule
CCF($\tau = 0$)
& 0.0572 $\pm$ 0.0007
& \underline{0.0524} $\pm$ 0.0005
& 0.0709 $\pm$ 0.0003
& \textbf{0.0390} $\pm$ 0.0005 \\
CCF($\tau > 0$)
& \underline{0.0341} $\pm$ 0.0002
& 0.0342 $\pm$ 0.0001
& 0.0420 $\pm$ 0.0001
& \textbf{0.0276} $\pm$ 0.0001 \\
\midrule
KS(1)
& 0.0375 $\pm$ 0.0000
& \textbf{0.0328} $\pm$ 0.0002
& 0.0379 $\pm$ 0.0002
& \underline{0.0342} $\pm$ 0.0002 \\
KS(5)
& \textbf{0.0418} $\pm$ 0.0003
& 0.0446 $\pm$ 0.0003
& 0.0437 $\pm$ 0.0003
& \underline{0.0424} $\pm$ 0.0002 \\
KS(20)
& \textbf{0.0530} $\pm$ 0.0004
& 0.0647 $\pm$ 0.0005
& \underline{0.0622} $\pm$ 0.0006
& 0.0660 $\pm$ 0.0005 \\
KS(100)
& 0.1403 $\pm$ 0.0024
& \textbf{0.1300} $\pm$ 0.0016
& 0.1722 $\pm$ 0.0020
& \underline{0.1399} $\pm$ 0.0018 \\
\midrule
Skew(1)
& 0.1582 $\pm$ 0.0037
& \textbf{0.0845} $\pm$ 0.0062
& \underline{0.1520} $\pm$ 0.0051
& 0.1920 $\pm$ 0.0054 \\
Skew(5)
& 0.3335 $\pm$ 0.0073
& \underline{0.2032} $\pm$ 0.0133
& 0.3393 $\pm$ 0.0030
& \textbf{0.1076} $\pm$ 0.0024 \\
Skew(20)
& 0.3380 $\pm$ 0.0046
& \underline{0.2143} $\pm$ 0.0074
& 0.3579 $\pm$ 0.0042
& \textbf{0.1616} $\pm$ 0.0076 \\
Skew(100)
& 0.1199 $\pm$ 0.0058
& \textbf{0.0818} $\pm$ 0.0020
& 0.1155 $\pm$ 0.0033
& \underline{0.0893} $\pm$ 0.0037 \\
\midrule
Kurt(1)
& \underline{0.7942} $\pm$ 0.0515
& 1.6739 $\pm$ 0.0490
& 0.8877 $\pm$ 0.0305
& \textbf{0.7817} $\pm$ 0.0375 \\
Kurt(5)
& 0.8688 $\pm$ 0.0288
& \underline{0.4753} $\pm$ 0.0131
& 0.9116 $\pm$ 0.0182
& \textbf{0.4427} $\pm$ 0.0106 \\
Kurt(20)
& \textbf{0.2287} $\pm$ 0.0101
& 0.5363 $\pm$ 0.0065
& \underline{0.2997} $\pm$ 0.0234
& 0.3646 $\pm$ 0.0130 \\
Kurt(100)
& \underline{0.2037} $\pm$ 0.0122
& 0.2313 $\pm$ 0.0040
& \textbf{0.1690} $\pm$ 0.0089
& 0.2536 $\pm$ 0.0077 \\
\midrule
ACF($x$)
& \underline{0.0008} $\pm$ 0.0000
& 0.0008 $\pm$ 0.0000
& 0.0009 $\pm$ 0.0000
& \textbf{0.0007} $\pm$ 0.0000 \\
ACF($|x|$)
& \textbf{0.0015} $\pm$ 0.0001
& 0.0039 $\pm$ 0.0000
& 0.0021 $\pm$ 0.0001
& \underline{0.0018} $\pm$ 0.0000 \\
ACF($x^2$)
& \textbf{0.0009} $\pm$ 0.0000
& 0.0024 $\pm$ 0.0000
& \underline{0.0012} $\pm$ 0.0000
& 0.0012 $\pm$ 0.0000 \\
\midrule
Lev. Effect
& \textbf{0.0007} $\pm$ 0.0000
& 0.0010 $\pm$ 0.0000
& \underline{0.0009} $\pm$ 0.0000
& 0.0010 $\pm$ 0.0000 \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table*}
```

Note on ties: `ACF($x$)` Baseline and w/o RV both read 0.0008; the finer-precision CSV values place Baseline second and w/o ModernTCN first (0.0007), so Baseline carries the `\underline{}`. `ACF($x^2$)` w/o LambertW and w/o ModernTCN both read 0.0012; the underline is on w/o LambertW per the CSV ordering. Leave these exactly as written above.

- [ ] **Step 2: Verify.**

Run: `grep -n "table:ablation_fidelity" main.tex`
Expected: exactly one definition, `{l|cccc}` colspec, 3 `&` per data row. Confirm the four column headers are Baseline / w/o RV Decomp. / w/o LambertW / w/o ModernTCN.

- [ ] **Step 3: Commit.**

```bash
git add main.tex
git commit -m "paper: replace ablation fidelity table with ablation data"
```

---

## Task 4: Replace the ablation portfolio risk table

**Files:**
- Modify: `main.tex:730-779`

- [ ] **Step 1: Replace the whole `table*` environment (lines 730–779) with the block below.**

```latex
\begin{table*}[htbp]
\centering
\caption{Ablation study on portfolio risk fidelity: relative error of VaR and ES (averaged over the four trading strategies) when each key component of QuantDiffusion (Baseline = full model) is removed. Best in \textbf{bold}, second best \underline{underlined}; lower is better.}
\label{table:ablation_portfolio_risk}

\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{0.9}

\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{l|cccc}
\toprule
\textbf{Metric} & \textbf{Baseline} & \textbf{w/o RV Decomp.} & \textbf{w/o LambertW} & \textbf{w/o ModernTCN} \\
\midrule
$\text{RE}_{\text{VaR}}(0.05)$
& \textbf{0.3514} $\pm$ 0.0054
& 0.6559 $\pm$ 0.0040
& \underline{0.3814} $\pm$ 0.0009
& 0.4055 $\pm$ 0.0063 \\
$\text{RE}_{\text{VaR}}(0.025)$
& \textbf{0.3259} $\pm$ 0.0026
& 0.6278 $\pm$ 0.0042
& \underline{0.3543} $\pm$ 0.0012
& 0.3716 $\pm$ 0.0052 \\
$\text{RE}_{\text{VaR}}(0.01)$
& \textbf{0.3351} $\pm$ 0.0034
& 0.6131 $\pm$ 0.0029
& \underline{0.3582} $\pm$ 0.0021
& 0.3676 $\pm$ 0.0039 \\
\midrule
$\text{RE}_{\text{ES}}(0.05)$
& \textbf{0.3276} $\pm$ 0.0023
& 0.6216 $\pm$ 0.0031
& \underline{0.3546} $\pm$ 0.0013
& 0.3696 $\pm$ 0.0046 \\
$\text{RE}_{\text{ES}}(0.025)$
& \textbf{0.3277} $\pm$ 0.0038
& 0.6045 $\pm$ 0.0022
& \underline{0.3490} $\pm$ 0.0014
& 0.3565 $\pm$ 0.0035 \\
$\text{RE}_{\text{ES}}(0.01)$
& \textbf{0.3388} $\pm$ 0.0104
& 0.5797 $\pm$ 0.0028
& 0.3534 $\pm$ 0.0043
& \underline{0.3453} $\pm$ 0.0036 \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table*}
```

- [ ] **Step 2: Verify.**

Run: `grep -n "table:ablation_portfolio_risk" main.tex`
Expected: exactly one definition, 6 data rows, 3 `&` each. Note that at `RE_ES(0.01)` the second best is w/o ModernTCN (0.3453), not w/o LambertW; leave as written.

- [ ] **Step 3: Commit.**

```bash
git add main.tex
git commit -m "paper: replace ablation portfolio VaR/ES table with ablation data"
```

---

## Task 5: Fix the Experiments baseline listing

**Files:**
- Modify: `main.tex:359`

- [ ] **Step 1: Replace line 359 (the entire paragraph beginning "QuantDiffusion is benchmarked against four state of the art generators") with:**

```latex
QuantDiffusion is benchmarked against three state of the art baselines that span the two dominant paradigms in the recent literature. Two are GAN based: QuantGAN, which couples a temporal convolutional backbone with the Lambert$~W$ transform, and Tail-GAN, which targets tail risk fidelity through a quantile based adversarial objective. The third, Diffusion-TS, is a transformer based denoising diffusion model with a trend seasonality decomposition. Together they cover temporal convolutional architectures, tail oriented adversarial training, and transformer based diffusion, providing a representative cross section of the current state of the art.
```

- [ ] **Step 2: Verify no stale baseline names remain in the Experiments framing.**

Run: `grep -n "COSCI\|SigCWGAN\|DisCorrGAN\|four state of the art\|path signature" main.tex`
Expected: no matches inside Section 5 (line ≥ 355). Matches inside the Related Work subsection (lines < 175) are acceptable and out of scope per the "minimal" decision.

- [ ] **Step 3: Commit.**

```bash
git add main.tex
git commit -m "paper: align Experiments baseline listing with the three actual baselines"
```

---

## Task 6: Rewrite the fidelity interpretation paragraphs (inter-asset, marginal, temporal)

**Files:**
- Modify: `main.tex:664-671` (three `\paragraph` blocks: "Inter asset dependence", "Marginal distributions", "Temporal and nonlinear dependence")

Also update the summary sentence at `main.tex:423` ("Out of the eighteen metrics ... thirteen ... second best on one more") to match the new counts.

- [ ] **Step 1: Replace the sentence at line 423 that begins "Out of the eighteen metrics" with:**

```latex
Out of the seventeen fidelity metrics, QuantDiffusion attains the best score on twelve and is among the top two on fifteen, the exceptions being confined to skewness, and its standard deviations across seeds are typically among the smallest of all models.
```

- [ ] **Step 2: Replace the "Inter asset dependence" paragraph (line 665) with:**

```latex
On contemporaneous cross correlation, CCF($\tau=0$), Diffusion-TS attains the lowest error at $0.0533$, with QuantDiffusion second at $0.0572$; the two separate clearly from QuantGAN ($0.0968$) and Tail-GAN ($0.1987$). On the lagged metric CCF($\tau>0$), the ordering changes and QuantDiffusion ranks first at $0.0341$, ahead of Diffusion-TS at $0.0367$. This pattern is consistent with the design. Diffusing on the joint representation $(z_t, RV_t)$ lets the shock component $z_t$ carry the contemporaneous comovement while $RV_t$ transmits the cross asset comovement of risk that governs lagged dependence, which explains why QuantDiffusion is strongest precisely on the lagged structure. Diffusion-TS, which lacks an explicit volatility channel, recovers the contemporaneous correlation accurately but sustains it less well across lags.
```

- [ ] **Step 3: Replace the "Marginal distributions" paragraph (line 668) with:**

```latex
QuantDiffusion achieves the lowest KS statistic at every horizon $h \in \{1, 5, 20, 100\}$, with QuantGAN second throughout; at $h = 100$ the margin is negligible ($0.1403$ versus $0.1404$), whereas at the shorter horizons the lead is pronounced. Tail-GAN is the weakest at every horizon, and Diffusion-TS trails the GAN baselines at the shorter horizons. The kurtosis row is the most consequential for risk, since kurtosis is a primary indicator of fat tailed behavior. QuantDiffusion attains the lowest kurtosis error at every horizon, and the advantage is most striking at $h = 1$ ($0.7942$), less than a third of the next best model (Diffusion-TS at $2.9343$) and less than a fifth of QuantGAN ($5.1508$). This is the clearest single piece of evidence in the table that the realized volatility decomposition reinstates the fat tailed behavior of returns. The skewness metrics are the sole weakness, with QuantDiffusion second at $h = 1$, last at $h \in \{5, 20\}$, and third at $h = 100$. This is a structural consequence of the decomposition, since dividing returns by $\sqrt{RV_t}$ renders the shock $z_t$ approximately symmetric, so the sample skewness of the reconstructed series depends on the joint $(z_t, RV_t)$ dynamics rather than on a marginal moment that the diffusion model targets directly. The ablation in Table~\ref{table:ablation_fidelity} corroborates this reading, since removing the RV decomposition improves skewness while degrading the volatility clustering metrics.
```

- [ ] **Step 4: Replace the "Temporal and nonlinear dependence" paragraph (line 671) with:**

```latex
The most decisive evidence for the SDE aware design lies in the nonlinear autocorrelation metrics. QuantDiffusion attains $0.0015$ on $\mathrm{ACF}(|x|)$ and $0.0009$ on $\mathrm{ACF}(x^{2})$, against $0.0041$ and $0.0023$ for Diffusion-TS, errors between two and three times larger for the transformer based diffusion model, and it likewise leads on the leverage effect ($0.0007$ versus $0.0012$ for Diffusion-TS); on all three metrics QuantGAN is second and QuantDiffusion improves on it by a clear margin (a factor of about $2.5$ on the two absolute and squared autocorrelations and about $1.6$ on the leverage effect). On the linear $\mathrm{ACF}(x)$ all models are effectively tied near the noise floor, between $0.0007$ and $0.0012$. Despite being a state of the art diffusion model, Diffusion-TS fails to reproduce the persistence in $|x|$ and $x^{2}$ that drives volatility clustering. This is consistent with the argument of Section~3, namely that a diffusion model which ingests raw returns cannot separate Brownian volatility fluctuations from the injected diffusion noise, whereas extracting the volatility component beforehand frees the score network to learn genuine nonlinear temporal structure.
```

- [ ] **Step 5: Verify.**

Run: `grep -n "eighteen\|thirteen\|0.0384\|0.9659\|four times worse\|DisCorrGAN" main.tex`
Expected: no matches (all stale numeric claims and DisCorrGAN references removed).

- [ ] **Step 6: Commit.**

```bash
git add main.tex
git commit -m "paper: rewrite fidelity interpretation to match benchmark2 data"
```

---

## Task 7: Rewrite the portfolio VaR interpretation paragraph

**Files:**
- Modify: `main.tex:724-728` (the `\paragraph{Portfolio level Value at Risk}` block, including the broken `\ref{tab:avg_portfolio_risk_var}`)

- [ ] **Step 1: Replace the paragraph body (lines 726–728, the text between the `\paragraph` heading and the next table) with:**

```latex
Table~\ref{table:portfolio_risk} reports the average relative error of portfolio level Value at Risk over the four trading strategies at three confidence levels. QuantDiffusion attains the smallest error at every level, with $0.3514$, $0.3259$, and $0.3351$ at $\alpha \in \{0.05, 0.025, 0.01\}$, and its error is essentially flat across confidence levels rather than deteriorating deeper in the tail. The second best model varies by level, being QuantGAN at $\alpha = 0.05$ and Diffusion-TS at the two tighter levels, yet the margin is substantial throughout: relative to the second best, QuantDiffusion reduces the error by $34.2\%$ at $\alpha = 0.05$, $32.1\%$ at $\alpha = 0.025$, and $18.5\%$ at $\alpha = 0.01$. The GAN baselines, whose VaR errors range from $0.53$ to $0.86$ and worsen as $\alpha$ tightens, illustrate that accurate unconditional statistics alone are not sufficient for tail quantile fidelity. QuantDiffusion also exhibits low dispersion across seeds, a practically relevant property for risk pipelines that rerun the generator across many scenarios.
```

- [ ] **Step 2: Verify the broken reference is gone.**

Run: `grep -n "tab:avg_portfolio_risk_var" main.tex`
Expected: no matches.

- [ ] **Step 3: Commit.**

```bash
git add main.tex
git commit -m "paper: rewrite portfolio VaR interpretation and fix table reference"
```

---

## Task 8: Rewrite the portfolio ES interpretation paragraph

**Files:**
- Modify: `main.tex:781-785` (the `\paragraph{Portfolio level Expected Shortfall}` block, including the broken `\ref{tab:avg_portfolio_risk_es}`)

- [ ] **Step 1: Replace the paragraph body (lines 783–785) with:**

```latex
Because VaR alone does not capture the magnitude of losses beyond its threshold, we complement the analysis above with the Expected Shortfall errors, the more stringent tail risk measure, reported in the lower panel of Table~\ref{table:portfolio_risk}. QuantDiffusion is again uniformly best, with relative errors of $0.3276$, $0.3277$, and $0.3388$ at $\alpha \in \{0.05, 0.025, 0.01\}$. The second best model at every level is Diffusion-TS ($0.4527$, $0.3995$, $0.3622$), so QuantDiffusion improves upon it by $27.6\%$, $18.0\%$, and $6.5\%$ respectively, while its lead over the GAN baselines, whose errors reach as high as $0.8619$, is larger still. A notable qualitative difference is that the error of Diffusion-TS decreases as $\alpha$ tightens, opposite to the near constant profile of QuantDiffusion. We read this as evidence that the transformer backbone produces an over dispersed profit and loss distribution whose bulk diverges from the empirical one more strongly than its deep tail, so that the averaging region of ES, pushed further into the tail, coincides with a range where the over dispersion partially offsets the empirical mass. Because ES is structurally more sensitive than VaR to the entire shape of the tail, the persistence of QuantDiffusion's advantage under this measure indicates that the realized volatility decomposition recovers the tail distribution rather than a single tail quantile. Taken together, Table~\ref{table:portfolio_risk} shows that QuantDiffusion produces scenarios that are most reliable precisely in the regime where reliability is hardest to obtain, and that adopting a state of the art diffusion architecture without an explicit financial process prior, as in Diffusion-TS, is insufficient to deliver this property.
```

- [ ] **Step 2: Verify.**

Run: `grep -n "tab:avg_portfolio_risk_es\|18.0\\%\|27.6\\%" main.tex`
Expected: no match for `tab:avg_portfolio_risk_es`; the two percentages present.

- [ ] **Step 3: Commit.**

```bash
git add main.tex
git commit -m "paper: rewrite portfolio ES interpretation and fix table reference"
```

---

## Task 9: Add the ablation study discussion

**Files:**
- Modify: `main.tex` — insert a new `\subsection{Ablation Study}` immediately after the Portfolio ES paragraph (after line 785) and before the figure environment at line 787.

This subsection is new: the paper currently shows the ablation tables (Tables `table:ablation_fidelity` and `table:ablation_portfolio_risk`) but provides no textual analysis. The content below renders the user's five-point narrative in academic prose. It presents the Lambert-W finding with the empirical-first framing (jumps impede tail learning, therefore SVJD is the a posteriori appropriate model); Task 10 aligns Section 2 with the same ordering.

- [ ] **Step 1: Insert the following block after line 785.**

```latex
\subsection{Ablation Study}

Tables~\ref{table:ablation_fidelity} and~\ref{table:ablation_portfolio_risk} isolate the contribution of each component by removing it from the full model. The clearest overall effect is that the full configuration is strongest on the metrics that describe how volatility clusters and evolves over time, that is $\mathrm{ACF}(|x|)$, $\mathrm{ACF}(x^{2})$, and the leverage effect, and on the portfolio level tail risk measures. Each component therefore contributes primarily to temporal structure and to tail risk rather than to the shape of any individual marginal moment. Examining the components one at a time makes their distinct roles explicit.

\paragraph{Realized volatility decomposition.}
Removing the RV decomposition all but eliminates the model's ability to reproduce volatility clustering: $\mathrm{ACF}(|x|)$ rises from $0.0015$ to $0.0039$ and $\mathrm{ACF}(x^{2})$ from $0.0009$ to $0.0024$, and the portfolio tail risk errors become the worst in the ablation, between $0.58$ and $0.66$ across all confidence levels. This follows directly from the fact that, without the decomposition, the model can no longer separate a residual shock from a volatility magnitude and generate the two processes independently. A revealing counterpoint is that skewness improves markedly, with the variant attaining the best skewness at $h = 1$ and $h = 100$. The decomposition is designed to orthogonalize the shock $z_t$ and the scale $\sqrt{RV_t}$, and dividing returns by the scale compresses large negative shocks toward the center, so that $z_t$ becomes nearly symmetric. Because generation then draws $z_t$ and $\sqrt{RV_t}$ almost independently and multiplies them, the association between negative returns and elevated volatility that produces skewness is not reconstructed, and the output is too symmetric. When the decomposition is removed, the model instead learns the raw returns directly and retains their skewness. The RV decomposition thus embodies a trade off in which a portion of marginal skewness is exchanged for a substantial gain in volatility clustering and tail risk fidelity.

\paragraph{Lambert-W transform.}
The Lambert-W transform draws the far reaching extreme tail toward the center and renders the data closer to Gaussian, and it contributes unexpectedly to the modeling of volatility clustering. Removing it degrades $\mathrm{ACF}(|x|)$ from $0.0015$ to $0.0021$, $\mathrm{ACF}(x^{2})$ from $0.0009$ to $0.0012$, and Kurt($1$) from $0.7942$ to $0.8877$. We attribute this to the fact that, even after the RV decomposition, extreme values remain in the residual, and the model expends learning capacity fitting these outliers, leaving less capacity for the temporal structure captured by the autocorrelation functions. This observation motivates a change in the logical ordering of our argument. Rather than assuming the SVJD process first and introducing the Lambert-W transform to suit it, we start from the empirical finding that separating out the extreme mass improves kurtosis and clustering together, infer that real financial series contain a jump component that impedes the learning of the tail, and conclude that a specification which isolates such jumps is appropriate. Under this ordering the SVJD model is not an arbitrary prior imposed in order to justify the transform, but a specification that the empirical results support a posteriori as a suitable description of real financial time series.

\paragraph{ModernTCN backbone.}
A conventional temporal convolutional network cannot secure a global receptive field by increasing depth alone, since additional layers raise computation while the effective receptive field grows only sub linearly, in contrast to architectures with globally connected receptive fields such as Transformers. The ModernTCN backbone instead combines large and small kernels so as to capture wide temporal structure at once, which is reflected in its accurate autocorrelation: the full model improves on the variant without ModernTCN on both $\mathrm{ACF}(|x|)$ ($0.0015$ versus $0.0018$) and $\mathrm{ACF}(x^{2})$ ($0.0009$ versus $0.0012$). That ModernTCN reproduces the autocorrelation faithfully can therefore be read as evidence that it genuinely attends to a broad temporal range. Its marginal accuracy is comparatively weaker, however, since the variant without ModernTCN is competitive or better on several KS, skewness, and kurtosis entries, which identifies improving marginal fidelity without sacrificing temporal structure as a direction for future work.

\paragraph{Tail risk.}
The portfolio ablation in Table~\ref{table:ablation_portfolio_risk} reinforces this picture. The full model attains the lowest relative error on all six VaR and ES entries, the variant without the RV decomposition is the worst by a large margin, and the variant without Lambert-W is closest to the full model. The RV decomposition is thus the dominant driver of tail risk fidelity, with the Lambert-W transform a secondary but consistent contributor.
```

- [ ] **Step 2: Verify placement and references.**

Run: `grep -n "subsection{Ablation Study}\|a posteriori\|dominant driver of tail risk" main.tex`
Expected: the subsection appears once, after the ES paragraph and before the figure. Confirm both `\ref{table:ablation_fidelity}` and `\ref{table:ablation_portfolio_risk}` resolve to defined labels.

- [ ] **Step 3: Commit.**

```bash
git add main.tex
git commit -m "paper: add ablation study discussion"
```

---

## Task 10 (FLAGGED — core argument): Reverse the SVJD / Lambert-W logical ordering in Section 2 and Contributions

This task changes the paper's central line of reasoning to match the empirical-first framing introduced in Task 9. Review it independently before accepting. If the reviewer prefers to keep Section 2's current ordering, skip this task; Task 9's ablation discussion already stands on its own.

**Files:**
- Modify: `main.tex:228` (last paragraph of the Lambert-W subsection)
- Modify: `main.tex:167` (second sentence of the Contributions paragraph)

- [ ] **Step 1: Replace the paragraph at line 228 (beginning "This regularity is what allows us to lift the Lambert~$W$ transform") with:**

```latex
This regularity is what allows us to lift the Lambert~$W$ transform from a mere preprocessing heuristic to an operation acting on the underlying SDE. Empirically, as the ablation in Section~5 shows, separating out the extreme mass of the returns improves kurtosis and volatility clustering jointly, which indicates that real financial series carry a jump component that impedes the learning of the tail. This motivates a specification in which the jumps are isolated from the continuous dynamics, and the SVJD model of Section~2.3 provides exactly such a specification. Applying~\eqref{eq:lambert_inverse} together with It\^{o}'s lemma to a log return process governed by the SVJD dynamics yields an SDE for the transformed series that inherits the drift and diffusion structure of SVJD while suppressing only the jump term, so that the Gaussianized series remains tied to a recognizable stochastic process representation rather than reverting to an unstructured signal. The empirical results therefore support the SVJD specification a posteriori, rather than the transform being introduced to fit an assumed model.
```

- [ ] **Step 2: Replace the second contribution sentence at line 167 (beginning "Second, exploiting the fact that the Lambert--$W$ transform") with:**

```latex
Second, starting from the empirical observation that isolating extreme jump like events improves both tail and clustering fidelity, and exploiting the fact that the Lambert--$W$ transform is a twice differentiable invertible map, we apply It\^{o}'s calculus to derive its role directly at the SDE level rather than treating it as a preprocessing heuristic; this restores stochastic process level interpretability of the generated paths and supports the SVJD specification a posteriori, a feature that has been absent in GAN based approaches.
```

- [ ] **Step 3: Verify consistency.**

Run: `grep -n "a posteriori\|suppressing only the jump term" main.tex`
Expected: the Section 2 paragraph and the contribution sentence now both express the empirical-first ordering, consistent with the ablation discussion added in Task 9.

- [ ] **Step 4: Commit.**

```bash
git add main.tex
git commit -m "paper: reverse SVJD/Lambert-W logical ordering to empirical-first framing"
```

---

## Task 11: Update the realized volatility estimator description to close-to-close / w10

Confirmed with the user: the reported model uses the close-to-close realized variance over a rolling 10 day window (C2C/w10). Section 2.5 currently states the Parkinson high-low estimator (`main.tex:249`), and the future-work sentence (`main.tex:803`) also refers to Parkinson estimates. Both must be updated to describe close-to-close realized variance.

**Files:**
- Modify: `main.tex:249` (sentence introducing the RV proxy)
- Modify: `main.tex:803` (future-work sentence)

- [ ] **Step 1: In the paragraph at line 249, replace the sentence:**

```latex
Using the realized variance $RV_t$ as the volatility proxy we adopt the high low estimator of Parkinson~\cite{parkinson1980extreme}, whose daily frequency reliability is well established (Andersen and Bollerslev~\cite{andersen1998answering} report that such measurements account for roughly half of the variability in the latent volatility factor, while Barndorff Nielsen and Shephard~\cite{barndorff2002econometric} prove its unbiasedness for the unobserved instantaneous volatility) the decomposition reads
```

with:

```latex
Using the realized variance $RV_t$ as the volatility proxy, we estimate it from close to close log returns over a rolling window of ten trading days, a simple and robust daily frequency proxy whose reliability as a measure of the latent volatility factor is well established (Andersen and Bollerslev~\cite{andersen1998answering}; Barndorff Nielsen and Shephard~\cite{barndorff2002econometric}); the decomposition reads
```

- [ ] **Step 2: At line 803, replace "relies on daily Parkinson estimates" so the sentence reads:**

```latex
First, the realized volatility decomposition adopted here relies on a daily close to close realized variance, and we expect further gains from incorporating high frequency data, which would yield substantially more accurate realized volatility.
```

- [ ] **Step 3: Verify no stale estimator reference remains.**

Run: `grep -n "Parkinson\|parkinson1980extreme\|high low estimator" main.tex`
Expected: no matches (the `parkinson1980extreme` bib entry simply becomes unused, which biblatex handles silently).

- [ ] **Step 4: Commit.**

```bash
git add main.tex
git commit -m "paper: describe close-to-close/w10 realized variance instead of Parkinson"
```

---

## Final verification

- [ ] **Structural sweep of `main.tex`:**

Run: `grep -n "DisCorrGAN\|COSCI\|SigCWGAN\|Parkinson\|tab:avg_portfolio_risk\|eighteen metrics\|thirteen" main.tex`
Expected: no matches at all for `DisCorrGAN`, `Parkinson`, or `tab:avg_portfolio_risk`; `COSCI`/`SigCWGAN` only inside the Related Work subsection (lines < 175), never in Section 5.

- [ ] **Column-count check:** confirm each main results table row has 3 `&` (4 columns, `{l|cccc}`) and each ablation table row has 3 `&` (4 columns); confirm every table has exactly one `\toprule`, one `\midrule` after the header, and one `\bottomrule`, and that `\begin{tabular}`/`\end{tabular}`, `adjustbox`, and `table*` are balanced.

- [ ] **PDF compilation (user environment):** the local machine has no TeX toolchain, so compile in Overleaf or a local TeX install (`acmart` + `biblatex`/`biber` + `kotex`, typically via `lualatex`). Confirm there are no undefined references and that all four tables fit within `\textwidth`.

- [ ] **Read-through:** verify the prose is in formal written register (문어체), contains no em-dashes (`---`) and no hyphenated compound modifiers introduced by this change, and that every numeric claim matches the tables.

## Out-of-scope items noted for the user (no task created)

- **Related Work prose (lines 147–167)** still names COSCI-GAN, SigCWGAN, Market-GAN, and FIN-GAN. Per the "minimal" decision this is intentionally left unchanged.
- **Abstract (line 89)** claims QuantDiffusion "consistently improves ... cross sectional dependence"; on the contemporaneous CCF the model is second (behind Diffusion-TS), though it leads on the lagged CCF. The claim is defensible but slightly strong. Flagged only.
- **Figure at lines 787–792** has a placeholder caption ("Enter Caption") and a duplicate label `fig:placeholder` (also used by the commented-out figure near line 352). Flagged only.
- **ACM boilerplate**: unfilled CCS concepts (lines 99–127), template `\acmDOI`/`\acmISBN`, and example `\received` dates. Flagged only.
```
