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

---

## Assets

| Asset | Contracts | Active Period | Tick Size | 5-min Mean Range |
|---|---|---|---|---|
| **NQ** (Nasdaq, primary) | NQH20, NQM20 | Jan 2 – May 31, 2020 | $0.25 | 38 ticks |
| **GC** (Gold, secondary) | GCG24, GCJ24, GCM24, GCQ24 | Jan 2 – May 31, 2024 | $0.10 | ~12 ticks |

**Why NQ:** The Jan–May 2020 window spans pre-COVID, the March 2020 crash, and the recovery — providing exceptional regime heterogeneity (volatility swings from ~12 ticks to 1,000+ ticks per 5-min bar). This makes state-conditioning on the ePDF most informative. Data comes from stitching NQH20 (Jan–Mar 20) and NQM20 (Mar 20–May 31) at the volume-crossover rollover date.

**Date windows** are set to match the AIAgent coverage periods exactly. NQU20 rows outside Jan–May 2020 are excluded before contract stitching.

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
├── order_execution.ipynb          ← main notebook (all code)
├── README.md
├── data/
│   ├── Nasdaq/
│   │   ├── NQH20.csv
│   │   ├── NQM20.csv
│   │   ├── NQU20.csv
│   │   └── AIAgent_Nasdaq.csv     ← AI agent position signal (Jan–May 2020)
│   ├── Gold/
│   │   ├── GCG24.csv
│   │   ├── GCJ24.csv
│   │   ├── GCM24.csv
│   │   ├── GCQ24.csv
│   │   └── AIAgent_Gold.csv       ← AI agent position signal (Jan–May 2024)
│   └── ...
└── docs/
    └── specs/
        └── 2026-05-23-order-execution-design.md
```

### Running the notebook

```bash
cd monte-carlo-project-group
jupyter notebook order_execution.ipynb
```

Run all cells top to bottom. The notebook is self-contained and produces all results, tables, and figures.

---

## Configuration

All parameters live in **Section 0** (first code cell). Switching asset or holding period requires editing only this block.

```python
# --- Asset & Data ---
ASSET               = 'NQ'      # 'NQ' (Nasdaq) or 'GC' (Gold)
TICK_SIZE           = 0.25      # NQ=0.25, GC=0.10
DATA_DIR            = 'data/'
MIN_BARS_PER_DAY    = 350       # active-day filter (~90% of 390-min US session)

# --- Holding Period ---
TAU                 = 5         # minutes; valid: 5, 10, 15, 30, 60

# --- AI Agent Signal ---
USE_AI_AGENT_SIGNAL = True      # True = AIAgent CSV; False = synthetic momentum fallback

# --- EWMA State Engine ---
HALF_LIFE           = 20        # EWMA half-life in tau-bars
M_STATES            = 3         # volume regime bins  (low / medium / high)
N_STATES            = 3         # volatility regime bins
K_STATES            = 3         # direction bins      (down / flat / up)
DIR_THRESHOLD       = 1.0       # ticks; dead-band around zero for direction state

# --- ePDF Builder ---
BURN_IN             = 500       # tau-bars before ePDF estimation begins
MIN_EPDF_SAMPLES    = 30        # min observations per state before ePDF is trusted
SMOOTHING_ALPHA     = 0.5       # Laplace/Jeffreys smoothing pseudo-count per tick level

# --- Train / Test Split ---
TEST_START_FRAC     = 0.60      # first 60% = in-sample; remaining 40% = out-of-sample test

# --- Monte Carlo Simulation ---
N_MC_PATHS          = 10_000    # simulation paths per (state, direction) query
MC_CONFIDENCE       = 0.95      # CI level for fill-rate and improvement bands
BOOTSTRAP_REPS      = 1_000     # bootstrap resamples for ePDF estimation uncertainty

# --- Backtest & Order Placement ---
FILL_PROB_THRESHOLD = 0.60      # minimum fill probability to place a limit order
QUEUE_HAIRCUT       = 0.92      # queue-position discount on raw fill probability (1.0 = off)

# --- Synthetic Signal fallback (USE_AI_AGENT_SIGNAL = False only) ---
SIGNAL_TYPE         = 'momentum'
SIGNAL_LOOKBACK     = 10
SIGNAL_THRESHOLD    = 1.0

# --- Reproducibility ---
RANDOM_SEED         = 42
```

To switch to Gold futures:
```python
ASSET     = 'GC'
TICK_SIZE = 0.10
```

---

## Notebook Structure

| Section | Content |
|---|---|
| **0** | Configuration block |
| **1** | Imports and setup |
| **2** | Data loading — CSV parsing, Excel date decoding, per-contract date filtering, volume-crossover rollover stitching, active-day filtering |
| **3** | OHLC reconstruction — non-overlapping τ-min bars from 1-min rows, day-aligned windows |
| **4** | Range computation — R, R_up, R_dn in integer tick units; identity R = R_up + R_dn verified |
| **5** | EWMA state engine — Algorithm 1 (recursive, lag-1 out-of-sample), quantile-based state binning |
| **6** | ePDF builder — online conditional histograms, Laplace smoothing (α=0.5), three-tier fallback |
| **7** | Monte Carlo execution simulation — inverse-CDF path sampling, queue haircut, execution frontier, bootstrap CIs, optimal ℓ\* selection |
| **8** | AI Agent signal integration — position CSV parsing, order extraction, datetime alignment to OHLC bars, position overlay visualization |
| **9** | Historical backtest — out-of-sample limit order simulation, fill detection, missed-fill close-price fallback, `is_missed_limit` tracking |
| **10** | Results & analysis — full execution quality summary, regime breakdown, Figure 2 replication, gross vs net cumulative improvement, improvement distribution (filled vs missed), calibration scatter, sensitivity heatmaps, cross-asset comparison |

---

## Key Modeling Decisions

### Non-overlapping OHLC windows
Overlapping windows produce ~τ× more samples but adjacent bars share τ−1 minutes of data, introducing serial correlation that violates the i.i.d. assumption required for Monte Carlo sampling. Non-overlapping windows preserve independence at the cost of fewer training observations, offset by the conservative `BURN_IN = 500` bars.

### EWMA state engine — strict out-of-sample
Algorithm 1 from the project spec is implemented with a one-bar lag: the EWMA at bar `j` uses only `η_{j−1}`. Regime bin thresholds (tercile boundaries) are computed from EWMA history seen up to bar `j` only. This eliminates look-ahead bias in both the signal and the state classification.

### Sparse-state mitigation
With 27 state combinations (3×3×3), some cells accumulate few observations — especially extreme combinations like high-volume + low-volatility that were genuinely rare during the COVID period. Two mitigations apply: (1) Laplace/Jeffreys smoothing (α=0.5) prevents zero-probability tails; (2) a three-tier fallback hierarchy degrades gracefully from state ePDF → marginal ePDF → market order at open.

### Monte Carlo simulation with queue haircut
The execution frontier is estimated by sampling `N_MC_PATHS` future range realizations from the conditional ePDF. Fill rates are then discounted by `QUEUE_HAIRCUT` to account for queue position — orders at a given price level compete with others, so not every price touch results in a fill. `QUEUE_HAIRCUT = 0.92` reflects that on average ~8% of potential fills are lost to queue depth. The optimal limit level satisfies:

```
ℓ* = argmax_ℓ  E[improvement | state, ℓ]
     subject to: QUEUE_HAIRCUT × P(fill | state, ℓ) ≥ FILL_PROB_THRESHOLD
```

Bootstrap resampling (1,000 reps) quantifies ePDF estimation uncertainty separately from execution outcome variability.

### Missed-fill close-price fallback
When a placed limit order does not fill within the bar, the position is executed at the bar's close price — the practical worst-case cost of a missed limit in a single-period model. This ensures every trade signal results in an actual execution, producing an honest `is_missed_limit` flag and `improvement_ticks` that can be negative. The correct benchmark for strategy evaluation is the **net improvement vs the market order (open price)**, computed as `fill_rate × mean_improvement_filled`, because the close-price cost conflates execution quality with the bar's directional return.

### AI Agent signal
The `AIAgent_<Asset>.csv` records the agent's signed position size at each 5-minute bar. Orders are extracted as `order = position_t − position_{t−1}`: positive → Buy, negative → Sell, zero → Flat. The signal is sparse by design (~7% of bars are non-Flat), with mean order size of 3.55 contracts for NQ.

---

## Model Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Queue position | Fill probability overestimated at any given price level | `QUEUE_HAIRCUT = 0.92` discounts raw fill rates by 8% |
| OHLC-only path | Intra-bar price path unknown; a limit touched briefly may not fill | MC sampling from ePDF captures range variability implicitly |
| Single-period execution | Limits not filled within τ minutes fall back to bar close | `is_missed_limit` flag tracks all such events; cost reported separately |
| Close-price fallback bias | Missed-fill cost (-35 ticks avg) reflects bar directional return, not pure execution slippage | Report net vs open-price benchmark as primary metric |
| No partial fills | Orders are 100% filled or not filled | Standard assumption for OHLC-level analysis |
| No adverse selection | Fill at limit price assumed regardless of why price moved there | Documented; adverse selection is structurally present in missed-fill cost |
| i.i.d. bar assumption | MC sampling treats bars as independent draws | EWMA state-conditioning partially addresses volatility clustering |
| Static ePDF | Distributions not updated after test split | A rolling update would improve performance during regime shifts |

---

## Results Summary (NQ, τ = 5 min, April–May 2020 test period)

### Execution quality

| Metric | NQ | GC |
|---|---|---|
| Limit orders placed | 916 / 919 (99.7%) | 573 / 574 (99.8%) |
| **Limit fill rate** | **65.94%** | **68.24%** |
| Missed fill rate | 34.06% | 31.76% |
| Mean improvement \| filled | +3.32 ticks | +1.01 ticks |
| Mean cost \| missed fill (close) | −35.74 ticks | −15.57 ticks |
| **Net vs market order benchmark** | **+2.19 ticks** | **+0.69 ticks** |

The headline metric is **net vs market order benchmark** = `fill_rate × mean_improvement_filled`. This isolates genuine execution alpha from the directional bar return embedded in the close-price missed-fill cost. Both assets beat the market order benchmark on every regime state.

### Key findings

- **Regime differentiation is large:** ℓ\* spans 1 tick (low volatility, flat direction) to 47 ticks (high volatility, trending direction) — a 47× range. A regime-blind fixed limit would forfeit improvement in calm states or starve fill rates in volatile ones.

- **High-volatility states earn the highest fill rates:** vol=2 × sig=1 states achieve 70.1% fill (above the 60% constraint) because the ePDF's heavy right tail supports wider limit placement without sacrificing fill probability.

- **The COVID period provides essential training diversity:** Volatility states are heavily skewed toward high (44% of bars) due to the March 2020 crash and recovery, providing ePDF cells with rich observations across all regimes.

- **Sensitivity is stable:** Fill rates hold in the 58–68% range across τ = 5–30 min at the θ = 0.60 threshold. Mean improvement scales with τ as expected (+3.3 ticks at τ=5, +37.8 ticks at τ=30) because longer holding periods accommodate larger range excursions.

- **Cross-asset generalization:** The identical pipeline applied to GC (Gold, 2024) with only `ASSET` and `TICK_SIZE` changed produces coherent, asset-appropriate results — confirming the framework is fully generic.

---

## References

- Hirsa, A. — IEOR4703 Term Project 2 specification: *Volatility-Volume-based Order Management*
- Almgren, R. & Chriss, N. (2001) — Optimal execution of portfolio transactions
- Harris, L. (2003) — *Trading and Exchanges: Market Microstructure for Practitioners*

