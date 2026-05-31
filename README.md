# Volatility-Volume-based Order Management

**Columbia University — IEOR4703 Monte Carlo Simulation Methods (Hirsa)**  
**Term Project 2**  
**Authors:** Hanyu Zhang, Jingyao Lai

---

## Overview

This project implements a **volatility-volume-based limit order execution system** for futures markets. The central research question:

> *Given a trade decision to buy or sell, at what price level should we place a limit order — conditioned on the current market regime — to maximize expected execution quality?*

The system builds empirical conditional probability distributions of intra-period price range (R_up / R_dn) conditioned on a three-dimensional regime state (volume × volatility × direction). A **Monte Carlo simulation layer** propagates uncertainty through execution outcomes, producing fill-probability confidence bands and an execution frontier used to select the optimal limit level per regime.

**Trade signals come from a real AI Agent.** Each asset's `AIAgent_<Asset>.csv` records the agent's signed position size at every 5-minute bar. Order events (buy / sell / flat) are derived as `order = position_t − position_{t−1}`. The execution framework decides *how* to fill those orders efficiently — the project is about execution quality, not signal alpha.

**`order_execution_v2.ipynb` is the current authoritative notebook.** It corrects six bugs found in v1 (data leakage, wrong volatility proxy, row-count OHLC misalignment, wrong `delta_x` definition, implicit MC objective, ambiguous net metric). See the [v2 Corrections](#v2-corrections) section for details.

---

## Assets

| Asset | Contracts | Active Period | Tick Size | 5-min Mean Range |
|---|---|---|---|---|
| **Nasdaq** (primary) | NQH20, NQM20 | Jan 2 – May 31, 2020 | $0.25 | 38 ticks |
| **Gold** (secondary) | GCG24, GCJ24, GCM24, GCQ24 | Jan 2 – May 31, 2024 | $0.10 | ~12 ticks |
| **EuroStoxx** | VGH22, VGM22 | Mar – Jun 2022 | $0.50 | — |
| **GBP – British Pound** | BPM20, BPU20 | 2020 | auto-inferred | — |
| **German Bunds** | RXM25, RXU25, RXZ25 | 2025 | auto-inferred | — |
| **JPY – Japanese Yen** | JYU24, JYZ24 | 2024 | auto-inferred | — |
| **HeatingOil** | (see folder) | — | auto-inferred | — |

**Why Nasdaq:** The Jan–May 2020 window spans pre-COVID, the March 2020 crash, and the recovery — providing exceptional regime heterogeneity (volatility swings from ~12 ticks to 1,000+ ticks per 5-min bar). This makes state-conditioning on the ePDF most informative. Data comes from stitching NQH20 (Jan–Mar 20) and NQM20 (Mar 20–May 31) at the volume-crossover rollover date.

**Asset auto-discovery:** The v2 notebook scans `DATA_DIR` for subfolders, treats `AIAgent*.csv` as the signal file, and infers tick size from the price grid. Only `ASSET` (Section 0) needs to change between runs — set it to the **exact folder name** as printed by the Section 2 discovery cell. No paths to hand-edit. Tick sizes for Nasdaq and Gold are pinned via `TICK_OVERRIDE`; all other assets infer from the price grid.

---

## Setup

### Requirements

```bash
pip install numpy pandas matplotlib seaborn jupyter
```

Python 3.8+ required. No additional dependencies.

### Directory structure

```
monte-carlo-project-group/
├── order_execution_v2.ipynb       ← main notebook (v2, all fixes applied)
├── order_execution.ipynb          ← v1 (retained for comparison)
├── README.md
├── make_test_data.py              ← generates synthetic test assets (no real CSVs needed)
└── data/
    ├── Nasdaq/
    │   ├── NQH20.csv
    │   ├── NQM20.csv
    │   ├── NQU20.csv
    │   └── AIAgent_Nasdaq.csv         ← AI agent position signal (Jan–May 2020)
    ├── Gold/
    │   ├── GCG24.csv
    │   ├── GCJ24.csv
    │   ├── GCM24.csv
    │   ├── GCQ24.csv
    │   └── AIAgent_Gold.csv           ← AI agent position signal (Jan–May 2024)
    ├── EuroStoxx/
    │   ├── VGH22.csv
    │   ├── VGM22.csv
    │   └── AIAgent_EuroStoxx.csv
    ├── GBP – British Pound/
    │   ├── BPM20.csv
    │   ├── BPU20.csv
    │   └── AIAgent_GBPUSD.csv
    ├── German Bunds – German Government Bonds/
    │   ├── RXM25.csv
    │   ├── RXU25.csv
    │   ├── RXZ25.csv
    │   └── AIAgent_Bunds.csv
    ├── JPY – Japanese Yen/
    │   ├── JYU24.csv
    │   ├── JYZ24.csv
    │   └── AIAgent_JPY.csv
    └── HeatingOil/
        └── AIAgent_HeatingOil.csv
```

> **Note:** `DATA_DIR` in the v2 notebook defaults to `'../data/'`. If your notebook sits at the repo root (next to `data/`), change it to `'data/'`.

### Running the notebook

```bash
cd monte-carlo-project-group
jupyter notebook order_execution_v2.ipynb
```

Run all cells top to bottom. The notebook is self-contained and produces all results, tables, and figures.

---

## Configuration

All parameters live in **Section 0** (first code cell). Only `ASSET` (and occasionally `BURN_IN`) needs to change between asset runs.

```python
# --- Asset & Data ---
ASSET               = 'Nasdaq'    # folder name under DATA_DIR; use exact discovered name
                                  # TICK_SIZE is auto-resolved from ASSET_REGISTRY — do not set manually
DATA_DIR            = '../data/'  # path to the data/ folder; adjust if notebook is at repo root
MIN_BARS_PER_DAY    = 350         # active-day filter (~90% of 390-min US session)

# --- Holding Period ---
TAU                 = 5           # minutes; valid: 5, 10, 15, 30, 60

# --- AI Agent Signal ---
USE_AI_AGENT_SIGNAL = True        # True = AIAgent CSV; False = synthetic momentum fallback

# --- EWMA State Engine ---
HALF_LIFE           = 20          # EWMA half-life in tau-bars
M_STATES            = 3           # volume regime bins  (low / medium / high)
N_STATES            = 3           # volatility regime bins
K_STATES            = 3           # direction bins      (down / flat / up)
DIR_THRESHOLD       = 1.0         # ticks; dead-band around zero for direction state
VOL_PROXY           = 'ewmv'      # [FIX 1] 'ewmv' = EWM std of range (spec-correct)
                                  #          'range_mean' = old EWMA of range (v1 behaviour)

# --- ePDF Builder ---
BURN_IN             = 300         # tau-bars before ePDF estimation begins
MIN_EPDF_SAMPLES    = 30          # min observations per state before ePDF is trusted
SMOOTHING_ALPHA     = 0.5         # Laplace/Jeffreys smoothing pseudo-count per tick level

# --- Train / Test Split ---
TEST_START_FRAC     = 0.60        # first 60% = in-sample; remaining 40% = out-of-sample test

# --- Monte Carlo Simulation ---
N_MC_PATHS          = 10_000      # simulation paths per (state, direction) query
MC_CONFIDENCE       = 0.95        # CI level for fill-rate and improvement bands
BOOTSTRAP_REPS      = 1_000       # bootstrap resamples for ePDF estimation uncertainty
MC_OBJECTIVE        = 'expected_ticks'  # [FIX 3] 'expected_ticks' = argmax(ℓ × fill_rate)
                                        #          'cond_on_fill'   = widest feasible ℓ

# --- Backtest & Order Placement ---
FILL_PROB_THRESHOLD = 0.60        # minimum fill probability to place a limit order
QUEUE_HAIRCUT       = 0.92        # queue-position discount on raw fill probability (1.0 = off)

# --- OHLC Reconstruction ---
MIN_COMPLETENESS    = 0.6         # [FIX 4] drop tau-windows with < 60% of expected 1-min rows

# --- Synthetic Signal fallback (USE_AI_AGENT_SIGNAL = False only) ---
SIGNAL_TYPE         = 'momentum'
SIGNAL_LOOKBACK     = 10
SIGNAL_THRESHOLD    = 1.0

# --- Reproducibility ---
RANDOM_SEED         = 42
```

**`TICK_SIZE` is no longer a user-settable parameter.** It is auto-resolved from `ASSET_REGISTRY` in Section 2 — setting it manually was a latent bug in v1 (wrong tick if you forgot to change it when switching assets). To override for unlisted assets, use `TICK_OVERRIDE` in Section 2.

To switch to Gold futures, change only:
```python
ASSET = 'Gold'
```

---

## Notebook Structure

| Section | Content |
|---|---|
| **0** | Configuration block |
| **1** | Imports and setup |
| **2** | Data loading — asset auto-discovery, CSV parsing, Excel date decoding, per-contract date filtering, volume-crossover rollover stitching, active-day filtering; `TICK_SIZE` resolved from registry |
| **3** | OHLC reconstruction — wall-clock `resample(τ min)` with `MIN_COMPLETENESS` filter [FIX 4]; grid alignment verified |
| **4** | Range computation — R, R_up, R_dn in integer tick units; `delta_x = open.diff()` (open-to-open) [FIX 6]; identity R = R_up + R_dn verified |
| **5** | EWMA state engine — Algorithm 1 (recursive, lag-1 out-of-sample), EWM std volatility proxy [FIX 1], quantile-based state binning |
| **6** | ePDF builder — online conditional histograms trained on in-sample bars only [FIX 2], Laplace smoothing (α=0.5), three-tier fallback; leakage check printed |
| **7** | Monte Carlo execution simulation — inverse-CDF path sampling, queue haircut, execution frontier, configurable `MC_OBJECTIVE` [FIX 3], vectorized bootstrap CIs, optimal ℓ\* selection |
| **8** | AI Agent signal integration — position CSV parsing, order extraction, datetime alignment to OHLC bars, position overlay visualization |
| **9** | Historical backtest — out-of-sample limit order simulation, fill detection, missed-fill close-price fallback, `is_missed_limit` tracking |
| **10a** | Execution quality summary — separate `net_vs_market` and `realized_avg_all` metrics [FIX 5], regime breakdown |
| **10b** | ePDF visualization — Figure 2 replication, conditional PMF heatmaps |
| **10c** | Calibration & time-series — gross vs net cumulative improvement, rolling fill rate |
| **10d** | Sensitivity analysis — τ × threshold heatmaps of `net_vs_market`; bootstrap skipped for speed |
| **10e** | Cross-asset comparison — dynamically selects NQ/GC (real data) or synthetic pair |

---

## v2 Corrections

Six bugs from v1 were identified and fixed. All fixes are verified by inline checks printed at runtime.

| Fix | Section | Description | Runtime check |
|---|---|---|---|
| **FIX 1** | §5 | Volatility proxy was EWMA of range (a mean). Spec requires EWM **std** of range. Using the mean systematically under-represents volatility clustering. | Flip `VOL_PROXY` → volatility-state mix shifts |
| **FIX 2** *(critical)* | §6 | ePDF store was trained on **all** bars, including the test period — data leakage. Results in v1 were optimistic. | `[FIX 2 check] in-sample valid bars == store total  (MATCH — no leakage)` |
| **FIX 3** | §7 | MC optimization objective was implicit (`expected_ticks` hardcoded). Now explicit via `MC_OBJECTIVE`; bootstrap loop vectorized. | Flip `MC_OBJECTIVE` → ℓ\* table may change |
| **FIX 4** | §3 | OHLC windows were row-count-based (`range(0, n, tau)`). Missing 1-min rows caused bars to span >τ minutes and boundaries to drift. Now uses wall-clock `resample()`. | `[FIX 4 check] off the τ-min grid: 0` |
| **FIX 5** | §10a | Single "net improvement" figure averaged all limit orders including missed fills, conflating two distinct quantities. Now separately reports `net_vs_market` and `realized_avg_all`. | Two distinct lines in Section 10a output |
| **FIX 6** | §4 | Direction state used `close − open` (intrabar). Spec requires `open_j − open_{j-1}` (open-to-open). Old value preserved as `delta_x_intrabar` for synthetic signal only. | Column `delta_x` is now open-to-open |

> **v2 vs v1 results:** Expect v2 numbers to be slightly less optimistic than v1 for the same asset. The difference is the leakage correction (FIX 2) — not a regression. The corrected numbers are the trustworthy ones.

---

## Key Modeling Decisions

### OHLC reconstruction — wall-clock alignment (FIX 4)
v2 builds tau-minute bars using `pandas.resample(f'{tau}min', origin='start_day')` with a `MIN_COMPLETENESS = 0.6` filter that drops any window with fewer than 60% of its expected 1-min rows. This guarantees bar boundaries stay on the tau-minute grid regardless of data gaps or rollover discontinuities. v1's row-count method (`range(0, n, tau)`) drifted silently when rows were missing.

### Direction signal — open-to-open (FIX 6)
`delta_x = open_j − open_{j-1}` (open-to-open across consecutive bars). This matches the spec's definition of the directional carry between bars. v1 used `close − open` (intrabar), which conflates the direction signal with within-bar noise. The intrabar value is retained as `delta_x_intrabar` for the synthetic momentum fallback signal only.

### Volatility proxy — EWM std of range (FIX 1)
The volatility signal for `sig_state` binning is the EWM **standard deviation** of range (`ewmstd_rng`), controlled by `VOL_PROXY = 'ewmv'`. This measures how much range varies around its recent level — the correct notion of volatility-of-volatility. v1 used the EWMA of range itself (a level), which is insensitive to sudden volatility spikes unless the mean shifts.

### ePDF training window — in-sample only (FIX 2)
The ePDF store is populated over `range(burn_in, test_start_idx)` exclusively. No test-period bars enter the distributions. A runtime check confirms `store total == in-sample valid bars`. v1 trained the store over all bars, meaning the conditional distributions had already "seen" the bars being tested — producing inflated out-of-sample fill rates.

### EWMA state engine — strict out-of-sample
Algorithm 1 from the project spec is implemented with a one-bar lag: the EWMA at bar `j` uses only `η_{j−1}`. Regime bin thresholds (tercile boundaries) are computed from EWMA history seen up to bar `j` only. This eliminates look-ahead bias in both the volatility signal and the state classification.

### Monte Carlo objective — explicit and configurable (FIX 3)
The optimization selects the limit level that maximizes expected improvement subject to the fill-probability constraint. `MC_OBJECTIVE = 'expected_ticks'` uses `argmax(ℓ × fill_rate)` over feasible levels; `'cond_on_fill'` selects the widest feasible level. The bootstrap loop over estimation uncertainty is vectorized (no double Python loop) and can be skipped with `BOOTSTRAP_REPS = 0` for speed in sensitivity sweeps.

### Monte Carlo simulation with queue haircut
The execution frontier is estimated by sampling `N_MC_PATHS` future range realizations from the conditional ePDF. Fill rates are then discounted by `QUEUE_HAIRCUT` to account for queue position — orders at a given price level compete with others, so not every price touch results in a fill. `QUEUE_HAIRCUT = 0.92` reflects that on average ~8% of potential fills are lost to queue depth. The optimal limit level satisfies:

```
ℓ* = argmax_ℓ  E[improvement | state, ℓ]
     subject to: QUEUE_HAIRCUT × P(fill | state, ℓ) ≥ FILL_PROB_THRESHOLD
```

Bootstrap resampling (1,000 reps) quantifies ePDF estimation uncertainty separately from execution outcome variability.

### Headline metric — net vs market order (FIX 5)
Two distinct net metrics are reported:

- **`net_vs_market`** = `fill_rate × mean_improvement | filled` — the true execution alpha vs a market order at open price. This is the primary metric.
- **`realized_avg_all`** = mean improvement across all limit orders including missed-fill close-price fallbacks. This is the P&L-relevant number but conflates execution quality with the bar's directional return.

The close-price missed-fill cost (−35 ticks avg for NQ) is not pure execution slippage — it includes the full directional bar return on bars where the limit didn't fill. Reporting only `realized_avg_all` would make the strategy appear to lose money when it is in fact generating positive execution alpha.

### Missed-fill close-price fallback
When a placed limit order does not fill within the bar, the position is executed at the bar's close price — the practical worst-case cost of a missed limit in a single-period model. This ensures every trade signal results in an actual execution, producing an honest `is_missed_limit` flag and `improvement_ticks` that can be negative.

### Sparse-state mitigation
With 27 state combinations (3×3×3), some cells accumulate few observations — especially extreme combinations like high-volume + low-volatility that were genuinely rare during the COVID period. Two mitigations apply: (1) Laplace/Jeffreys smoothing (α=0.5) prevents zero-probability tails; (2) a three-tier fallback hierarchy degrades gracefully from state ePDF → marginal ePDF → market order at open.

### AI Agent signal
The `AIAgent_<Asset>.csv` records the agent's signed position size at each 5-minute bar. Orders are extracted as `order = position_t − position_{t−1}`: positive → Buy, negative → Sell, zero → Flat. The signal is sparse by design (~7% of bars are non-Flat), with mean order size of 3.55 contracts for NQ.

---

## Model Limitations

| Limitation | Impact | Mitigation | v2 status |
|---|---|---|---|
| Queue position | Fill probability overestimated at any given price level | `QUEUE_HAIRCUT = 0.92` discounts raw fill rates by 8% | Applied |
| OHLC-only path | Intra-bar price path unknown; a limit touched briefly may not fill | MC sampling from ePDF captures range variability implicitly | Documented |
| Single-period execution | Limits not filled within τ minutes fall back to bar close | `is_missed_limit` flag tracks all such events; cost reported separately | Applied |
| Close-price fallback bias | Missed-fill cost (−35 ticks avg) reflects bar directional return, not pure execution slippage | `net_vs_market` is the primary metric; `realized_avg_all` reported separately [FIX 5] | Fixed |
| No partial fills | Orders are 100% filled or not filled | Standard assumption for OHLC-level analysis | Documented |
| No adverse selection | Fill at limit price assumed regardless of why price moved there | Adverse selection is structurally present in missed-fill cost | Documented |
| i.i.d. bar assumption | MC sampling treats bars as independent draws | EWMA state-conditioning partially addresses volatility clustering | Documented |
| Static ePDF | Distributions not updated after test split | A rolling update would improve performance during regime shifts | Documented |
| ePDF data leakage (v1) | Test bars contaminated training distributions | ePDF loop hard-stops at `test_start_idx` [FIX 2] | **Fixed** |
| Row-count OHLC drift (v1) | Bars spanned >τ min when rows were missing | Wall-clock `resample()` with completeness filter [FIX 4] | **Fixed** |
| Wrong volatility proxy (v1) | EWMA of range understated volatility clustering | EWM std of range as default [FIX 1] | **Fixed** |

---

## Results Summary (Nasdaq, τ = 5 min, April–May 2020 test period)

*These numbers are from v2 (post-leakage-correction). They are slightly lower than v1 — that is correct.*

### Execution quality

| Metric | Nasdaq | Gold |
|---|---|---|
| Limit orders placed | 916 / 919 (99.7%) | 573 / 574 (99.8%) |
| **Limit fill rate** | **65.94%** | **68.24%** |
| Missed fill rate | 34.06% | 31.76% |
| Mean improvement \| filled | +3.32 ticks | +1.01 ticks |
| Mean cost \| missed fill (close) | −35.74 ticks | −15.57 ticks |
| **`net_vs_market` (primary)** | **+2.19 ticks** | **+0.69 ticks** |
| `realized_avg_all` | −9.98 ticks | −4.26 ticks |

The headline metric is **`net_vs_market`** = `fill_rate × mean_improvement_filled`. This isolates genuine execution alpha from the directional bar return embedded in the close-price missed-fill cost. Both assets beat the market order benchmark on every regime state.

### Key findings

- **Regime differentiation is large:** ℓ\* spans 1 tick (low volatility, flat direction) to 47 ticks (high volatility, trending direction) — a 47× range. A regime-blind fixed limit would forfeit improvement in calm states or starve fill rates in volatile ones.

- **High-volatility states earn the highest fill rates:** vol=2 × sig=1 states achieve 70.1% fill (above the 60% constraint) because the ePDF's heavy right tail supports wider limit placement without sacrificing fill probability.

- **The COVID period provides essential training diversity:** Volatility states are heavily skewed toward high (44% of bars) due to the March 2020 crash and recovery, providing ePDF cells with rich observations across all regimes.

- **Sensitivity is stable:** Fill rates hold in the 58–68% range across τ = 5–30 min at the θ = 0.60 threshold. Mean improvement scales with τ as expected (+3.3 ticks at τ=5, +37.8 ticks at τ=30) because longer holding periods accommodate larger range excursions.

- **Cross-asset generalization:** The identical pipeline applied to Gold (2024) with only `ASSET` changed produces coherent, asset-appropriate results — confirming the framework is fully generic.

---

## Validation Guide (Gradual Asset Testing)

Run **one asset at a time**, checking the four spot-checks below before moving on. Only `ASSET` (and occasionally `BURN_IN`) changes between runs.

### Per-run spot-checks

| Fix | Where to look | "Correct" output |
|---|---|---|
| FIX 2 | Section 6 print | `in-sample valid bars == store total  (MATCH — no leakage)` |
| FIX 4 | Section 3 print | `off the τ-min grid: 0` — even with rollover gaps |
| FIX 5 | Section 10a | `net_vs_market` and `realized_avg_all` are **two distinct lines** |
| Tick size | Section 2 print | Inferred/override value matches the contract's real tick |
| FIX 1 | Section 5 | Flipping `VOL_PROXY` changes the volatility-state distribution |
| FIX 3 | Section 7 | Flipping `MC_OBJECTIVE` may shift the ℓ\* table |

> If `[FIX 2 check]` prints `in-sample valid bars = 0`: `BURN_IN` exceeds the in-sample window. Lower `BURN_IN` (thin assets) or lower `MIN_BARS_PER_DAY` if too many days were filtered out.

### Recommended asset order

#### Step 1 — EuroStoxx (smallest contract set, good first test)
```python
ASSET           = 'EuroStoxx'
BURN_IN         = 300
TEST_START_FRAC = 0.60
```
Tick auto-infers to 0.5. This asset is thin on tradable signals in the test window — expect many market fallbacks and only a handful of limit orders. That is real data; it confirms the fallback path works correctly.

#### Step 2 — Nasdaq (primary asset)
```python
ASSET           = 'Nasdaq'
BURN_IN         = 500          # 3 contracts, long window
TEST_START_FRAC = 0.60
```
`TICK_OVERRIDE` pins Nasdaq to 0.25. Compare headline numbers against the Results Summary above (`net_vs_market ≈ +2.19 ticks`). **v2 will be slightly less optimistic than v1** — that difference is the leakage correction (FIX 2), not a regression.

#### Step 3 — Gold (cross-asset validation, different tick)
```python
ASSET           = 'Gold'
BURN_IN         = 500
```
`TICK_OVERRIDE` pins Gold to 0.10. Four contracts; confirm rollover stitching in the load log (each contract hands off to the next).

#### Step 4 — Remaining assets (Bunds, JPY, GBP, HeatingOil)
```python
ASSET = 'German Bunds – German Government Bonds'   # use EXACT discovered name
```
These assets have **no `TICK_OVERRIDE`** — tick is inferred from the price grid. Check the printed `Tick size (inferred …)` line. Rates/FX futures have small ticks and inference can be fooled by stray prices; if it looks wrong, pin it in `TICK_OVERRIDE` and re-run. Copy the folder name verbatim from the Section 2 discovery printout (en-dashes are not regular hyphens).

### Synthetic data (no real CSVs needed)

To dry-run on a machine without the real data files:
```bash
python make_test_data.py .
```
This writes three discovery-compatible folders:

| Folder | Purpose |
|---|---|
| `SynthTiny` | Fast smoke test (~30 s end-to-end) |
| `SynthSmall` | Rollover + 3% missing minutes — stresses FIX 4 |
| `SynthCovid` | Volatility-regime spike — stresses ePDF conditioning |

```python
ASSET           = 'SynthTiny'
BURN_IN         = 100
TEST_START_FRAC = 0.5
```
Use synthetic assets to confirm the code runs; do not interpret the numbers as real results.

### Performance notes

The slowest cell is **Section 10d (sensitivity)** — it re-runs the full pipeline per `(τ, threshold)` pair. Default grids are trimmed (`TAU_GRID=[5,10,15]`, `THRESH_GRID=[0.5,0.6,0.7]`); widen to `[5,10,15,30]` / `[0.5,0.6,0.7,0.8]` for the final report. Sensitivity and cross-asset runs skip bootstrap (`bootstrap_reps=0`) since they only need ℓ\*. While iterating, you can also reduce `BOOTSTRAP_REPS` to ~200 in Section 0 — it only narrows the Section 7 CI band.

---

## References

- Hirsa, A. — IEOR4703 Term Project 2 specification: *Volatility-Volume-based Order Management*
- Almgren, R. & Chriss, N. (2001) — Optimal execution of portfolio transactions
- Harris, L. (2003) — *Trading and Exchanges: Market Microstructure for Practitioners*
