# Volatility-Volume-based Order Management

**Columbia University — IEOR4703 Monte Carlo Simulation Methods (Hirsa)**  
**Term Project 2**

Authors: Hanyu Zhang, Jingyao Lai

---

## Overview

This project implements a **volatility-volume-based limit order execution system** for futures markets. The central research question:

> *Given a trade decision to buy or sell, at what price level should we place a limit order — conditioned on the current market regime — to maximize expected execution quality?*

The system builds empirical conditional probability distributions (ePDFs) of intra-period price range (rangeUp / rangeDn) conditioned on a three-dimensional regime state (volume × volatility × direction). A **Monte Carlo simulation layer** propagates uncertainty through execution outcomes, producing fill-probability confidence bands and an execution frontier used to select the optimal limit level per state.

**The AI-agent signal is a testing vehicle only.** The project is about *how* to execute a decision efficiently, not about predicting price direction.

---

## Assets Supported

The pipeline auto-discovers all assets in `data/` — no code changes required to add a new asset.

| Asset | Folder | Contracts | Period | Tick Size |
|---|---|---|---|---|
| Nasdaq (primary) | `Nasdaq/` | NQH20, NQM20, NQU20 | Jan–May 2020 | 0.25 |
| Gold | `Gold/` | GCG24, GCJ24, GCM24, GCQ24 | 2023–2024 | 0.10 |
| EuroStoxx | `EuroStoxx/` | VGH22, … | varies | 0.50 |
| German Bunds | `Bunds/` | GBL… | varies | 0.01 |
| GBP/USD | `GBPUSD/` | 6BH…, … | varies | 0.01 |
| Japanese Yen | `JPY/` | 6JH…, … | varies | 0.005 |
| Heating Oil | `HeatingOil/` | HOF…, … | varies | 0.01 |

Tick sizes are pinned in `Config.tick_override`. For a new asset, add its folder under `data/`, add the pinned tick, and re-run — no other changes needed.

---

## Repository Layout

```
monte-carlo-project-group/
├── order_execution_v4_2.ipynb   ← clean submission notebook (imports packages)
├── order_execution.py           ← full pipeline engine (Config, AssetRegistry,
│                                   run_pipeline, cross_asset_table, OrderExecution)
├── order_execution_plots.py     ← plotting helpers (frontiers, Figure-2, P&L,
│                                   calibration, sensitivity heatmaps)
├── README.md
└── data/
    ├── Nasdaq/
    │   ├── NQH20.csv
    │   ├── NQM20.csv
    │   ├── NQU20.csv
    │   └── AIAgent_Nasdaq.csv   ← AI-agent signal (optional; triggers overlap fix)
    ├── Gold/
    │   ├── GCG24.csv
    │   └── ...
    ├── EuroStoxx/
    ├── Bunds/
    ├── GBPUSD/
    ├── JPY/
    └── HeatingOil/
```

Each asset folder may optionally contain an `AIAgent*.csv` file. If present, the pipeline clips the sample to the AI/OHLC overlap window before computing the train/test split (FIX 7 — see below).

---

## Setup

### Requirements

```bash
pip install numpy pandas matplotlib seaborn jupyter
```

Python 3.8+. No additional dependencies.

### Running the notebook

```bash
cd monte-carlo-project-group
jupyter notebook order_execution_v4_2.ipynb
```

Run all cells top to bottom. The notebook is self-contained: it imports `order_execution` and `order_execution_plots`, then produces all results, tables, and figures in a clean format.

---

## Configuration

All parameters are controlled by the `Config` dataclass defined in `order_execution.py`. The notebook instantiates it in the first cell:

```python
import order_execution as oe
import order_execution_plots as oep

cfg = oe.Config(
    data_dir        = 'data/',
    tau             = 5,              # holding period in minutes
    half_life       = 20,             # EWMA half-life in τ-bars
    n_states        = 3,              # volatility regime states
    m_states        = 3,              # volume regime states
    fill_prob_threshold = 0.60,       # minimum fill probability for a limit order
    n_mc_paths      = 10_000,         # MC paths per state/direction cell
    mc_objective    = 'expected_ticks',
    use_ai_agent_signal   = True,
    restrict_to_ai_overlap = True,    # core fix — see FIX 7 below
)
PRIMARY_ASSET = 'Nasdaq'
```

**Key Config parameters:**

| Parameter | Default | Description |
|---|---|---|
| `data_dir` | `'data/'` | Root folder for all asset sub-folders |
| `tau` | `5` | Holding period (bar width) in minutes |
| `half_life` | `20` | EWMA half-life in τ-bars |
| `m_states` | `3` | Volume quantile bins |
| `n_states` | `3` | Volatility quantile bins |
| `burn_in` | `500` | Bars discarded before ePDF starts accumulating |
| `test_start_frac` | `0.60` | In-sample fraction; OOS starts at bar `int(n × 0.60)` |
| `fill_prob_threshold` | `0.60` | θ: minimum P(fill) to place a limit (else market order) |
| `queue_haircut` | `0.92` | Conservative discount for queue-position uncertainty |
| `mc_objective` | `'expected_ticks'` | Objective for choosing limit level per state |
| `use_ai_agent_signal` | `True` | Use `AIAgent*.csv` if present in asset folder |
| `restrict_to_ai_overlap` | `True` | Clip OHLC to AI/OHLC date overlap before split (FIX 7) |

**MC objectives:**

| Objective | Formula | Behaviour |
|---|---|---|
| `expected_ticks` | `argmax_l  l × haircut × P(fill\|l)` | Maximises expected ticks; misses are costless |
| `cond_on_fill` | `argmax_l  l  s.t. P(fill) ≥ θ` | Widest feasible level given threshold |
| `net_of_miss` | `argmax_l  E[realised ticks]` | Charges missed-fill close-price cost; very conservative in high-vol regimes |

---

## Pipeline (corrected order)

```
load_data
  └─ build_ohlc            (wall-clock resample, MIN_COMPLETENESS filter)
       └─ compute_ranges   (R, R_up, R_dn in integer tick units)
            └─ compute_ewma_states  (Algorithm 1: lag-1, vol proxy = EWM std)
                 └─ generate_signals
                      └─ restrict_to_ai_overlap   ← FIX 7: must precede build_epdf
                           └─ build_epdf          (trains only on in-sample bars)
                                └─ build_mc_frontier_cache
                                     └─ run_backtest
                                          └─ compute_summary_metrics
```

The pipeline is exposed in two ways:

```python
# Single asset — returns full result dict including trades, ePDF store, MC cache
res = oe.run_pipeline('Nasdaq', cfg)

# All discovered assets — returns list of result dicts + formatted DataFrame
results, table = oe.cross_asset_table(cfg.discover(), cfg)
```

---

## Key Modeling Decisions

### Non-overlapping OHLC windows (FIX 4)
OHLC bars are built with `resample()` on wall-clock time (not a rolling window), with a `MIN_COMPLETENESS` filter that drops bars covering fewer than 60% of the expected 1-min ticks. Overlapping windows produce ~τ× more samples but introduce serial correlation (adjacent bars share τ−1 minutes), violating the i.i.d. assumption required for Monte Carlo sampling.

### EWMA state engine — strict out-of-sample (FIX 1)
Algorithm 1 from the project spec is implemented with a one-bar lag: at bar `j`, only information from bars `0..j−1` is used. The volatility proxy is the **EWM standard deviation of range** (`ewmv`), not the EWM mean — consistent with the project spec definition of volatility. Regime bin thresholds are also computed from the EWMA series up to bar `j` only.

### Signal-based open-to-open direction (FIX 6)
`delta_x = open.diff()` — the open-to-open move, not close-to-open. This correctly reflects information available at bar placement time.

### ePDF trained only on in-sample bars (FIX 2)
The empirical conditional histogram accumulates counts only for bars in `[burn_in, test_start_idx)`. OOS bars are used only for backtest evaluation, not ePDF estimation, preventing data leakage.

### AI/OHLC overlap clipping before train/test split (FIX 7 — core fix)
When an `AIAgent*.csv` is present, `restrict_to_ai_overlap()` clips the OHLC DataFrame to `[max(ohlc_start, ai_start), min(ohlc_end, ai_end)]` **before** `build_epdf()` computes `test_start_idx = int(n × 0.60)`. Without this, the bar-count split is anchored to the raw OHLC length — for Nasdaq, NQU20 data runs through September 2020 while the AI signal ends May 2020. A naïve split placed the OOS boundary at May 29, leaving only 19 signal events in the test set. With the fix applied, the OOS window starts April 2020 and captures all 918 signal events.

### Sparse-state mitigation
With 27 state combinations (3×3×3), some cells accumulate few observations. Two mitigations: (1) Laplace/Jeffreys smoothing (α=0.5) prevents zero-probability tails; (2) a three-tier fallback degrades gracefully from per-state ePDF → marginal ePDF → market order.

### Execution metrics (FIX 5)
Two metrics are reported separately to avoid conflating fill-quality with missed-fill costs:
- **`gross_edge`** = `fill_rate × mean_improvement | filled` — execution quality conditional on filling
- **`net_vs_market`** = `realized_avg_all` — average ticks vs open including missed-fill close costs

---

## Corrections from v1 → v4

| Fix | Description | Impact |
|---|---|---|
| **FIX 1** | Volatility proxy = EWM **std** of range (`ewmv`), not EWM mean | Correct regime state assignment |
| **FIX 2** | ePDF trains only on bars `[burn_in, test_start_idx)` | Eliminates OOS data leakage |
| **FIX 3** | Laplace smoothing (α=0.5) + three-tier fallback | Handles sparse states without zero-prob tails |
| **FIX 4** | OHLC built with wall-clock `resample()` + `MIN_COMPLETENESS` filter | Correct non-overlapping bars |
| **FIX 5** | Two separate metrics: `gross_edge` and `net_vs_market` | Cleaner separation of fill quality vs miss cost |
| **FIX 6** | `delta_x = open.diff()` (open-to-open), not close-to-open | Correct directional signal at placement time |
| **FIX 7** | `restrict_to_ai_overlap()` called before `build_epdf()` | OOS window covers full AI signal window (918 vs 19 events for Nasdaq) |

---

## Results Summary (τ = 5 min, `expected_ticks` objective)

Cross-asset results from `cross_asset_table()`:

| Asset | Tick | Limit trades | Fill rate | Impr\|filled | gross_edge | net_vs_market |
|---|---|---|---|---|---|---|
| Bunds | 0.010 | 8 | 50.00% | 1.00 | 0.50 | −0.38 |
| EuroStoxx | 0.500 | 48 | 52.08% | 1.00 | 0.52 | −3.65 |
| GBPUSD | 0.010 | 62 | 70.97% | 1.00 | 0.71 | −0.56 |
| Gold | 0.100 | 559 | 67.44% | 1.01 | 0.68 | −4.38 |
| HeatingOil | 0.010 | 117 | 55.56% | 14.83 | 8.24 | −14.19 |
| JPY | 0.005 | 163 | 65.64% | 1.00 | 0.66 | −1.34 |
| **Nasdaq** | 0.250 | **864** | **64.93%** | **3.14** | **2.04** | −10.14 |

Nasdaq primary-asset detail (τ=5 min, OOS Apr–May 2020):

- **918 OOS signal events** — all within the AI signal window (Jan–May 2020)
- **864 limit orders placed** (54 market fallbacks)
- **64.93% limit fill rate**; mean improvement conditional on fill: 3.14 ticks
- **gross_edge = +2.04 ticks** (fills only)
- High-volatility regimes (COVID period) produce wider range distributions: limits placed at larger offsets while maintaining the 60% fill threshold
- ePDF fill-probability predictions are well-calibrated against realized fill rates

---

## Model Limitations

| Limitation | Notes | v4 status |
|---|---|---|
| No queue position | Fill probability overestimated at the touch (many orders ahead) | Queue haircut (92%) applied as a partial correction |
| No partial fills | Orders are 100% filled or unfilled | Unchanged |
| No latency modeling | Assumes instantaneous placement at bar open | Unchanged |
| OHLC-only path | Intra-bar price path unknown; a limit touched briefly may not fill | Unchanged |
| No adverse selection | Fill assumed at limit price regardless of why price moved there | Unchanged |
| i.i.d. bar assumption | MC sampling treats bars as independent; EWMA state-conditioning partially addresses volatility clustering | Regime conditioning is the primary mitigation |
| Thin assets | Bunds (8 trades), EuroStoxx (48) — too few OOS events for reliable inference | Flagged in cross-asset table; primary results use Nasdaq |

---

## Adding a New Asset

1. Create `data/<AssetName>/` and drop in contract CSVs (format: `ExcelDate,Hour,Min,O,H,L,C,Volume`)
2. Optionally add an `AIAgent<AssetName>.csv` for a real signal (otherwise a synthetic momentum signal is used)
3. Add the tick size to `Config.tick_override` in `order_execution.py`
4. Re-run the notebook — `AssetRegistry` auto-discovers the new folder

---

## Notebook Structure (`order_execution_v4_2.ipynb`)

| Cell | Content |
|---|---|
| **1** | Imports: `order_execution as oe`, `order_execution_plots as oep` |
| **2** | Config instantiation (`Config(...)`) and `PRIMARY_ASSET` selection |
| **3** | Asset registry display — lists all auto-discovered assets and contracts |
| **4** | Single-asset `run_pipeline` call; prints execution summary |
| **5** | Execution frontiers plot (fill probability vs limit level, by vol state) |
| **6** | Optimal levels table (per-regime l*, market-order flags) |
| **7** | Figure-2 replication (ePDF counts, PMF, fill counts, survival function) |
| **8** | Cumulative improvement plot (gross vs net, rolling fill rate) |
| **9** | Fill probability calibration (predicted survival vs realized fill rate) |
| **10** | MC objective comparison (`expected_ticks` vs `cond_on_fill` vs `net_of_miss`) |
| **11** | Regime-level breakdown table (fill rate and net ticks per vol × sig state) |
| **12** | Sensitivity heatmaps (τ × fill threshold grid for fill rate, gross edge, net) |
| **13** | Sensitivity plot via `oep.plot_sensitivity()` |
| **14** | Cross-asset table via `oe.cross_asset_table()` |

---

## References

- Hirsa, A. — IEOR4703 Term Project 2 specification: *Volatility-Volume-based Order Management*
- Almgren, R. & Chriss, N. (2001) — Optimal execution of portfolio transactions
- Harris, L. (2003) — *Trading and Exchanges: Market Microstructure for Practitioners*
