# Volatility-Volume-Based Order Management

**Columbia University — IEOR4703 Monte Carlo Simulation Methods (Hirsa)**  
**Term Project 2**

Authors: Hanyu Zhang, Jingyao Lai

---

## Overview

This project implements a state-conditioned limit-order execution system for futures markets.

The research question is:

> Given an existing buy/sell decision, where should a limit order be placed so that execution quality is improved without creating excessive missed-fill risk?

The model does **not** try to predict whether the market will go up or down. Directional signals are treated as an input. The project focuses on the execution layer: conditional on a trade decision, it estimates how far price is likely to move within the next holding window and chooses a limit level accordingly.

For each asset, the pipeline:

1. builds non-overlapping OHLC bars from one-minute futures data;
2. computes upward and downward intra-bar ranges in tick units;
3. assigns each bar to a regime state based on volume, volatility, and recent direction;
4. estimates empirical conditional distributions of favorable price movement;
5. uses Monte Carlo simulation to build fill-probability frontiers; and
6. backtests the selected limit levels out of sample.

The AI-agent signal, when available, is used only as a source of trade decisions. It is not the main object of the project.

---

## Main Findings

For the Nasdaq primary asset at `tau = 5` minutes and fill threshold `theta = 0.60`, the baseline `expected_ticks` objective produces many limit orders and a solid gross fill improvement, but missed fills are costly:

| Objective | Limit orders | Market fallbacks | Fill rate | Gross edge | Net vs market |
|---|---:|---:|---:|---:|---:|
| `expected_ticks` | 864 | 54 | 64.9% | +2.04 ticks | -10.14 ticks |
| `cond_on_fill` | 864 | 54 | 64.9% | +2.04 ticks | -10.14 ticks |
| `net_of_miss` | 47 | 871 | 53.2% | +3.72 ticks | -4.21 ticks |

The key lesson is that maximizing `ell * P(fill)` can overstate execution quality because it treats missed fills as costless. The `net_of_miss` objective is more conservative: it chooses a limit order only when the estimated realized net improvement is positive after charging missed fills at the open-to-close move.

---

## Repository Layout

```text
monte-carlo-project-group/
├── order_execution_v8.ipynb       # clean submission notebook
├── order_execution.py             # execution engine and backtest pipeline
├── order_execution_plots.py       # plotting helpers
├── README.md
└── data/
    ├── Nasdaq/
    │   ├── NQH20.csv
    │   ├── NQM20.csv
    │   ├── NQU20.csv
    │   └── AIAgent_Nasdaq.csv     # optional external trade signal
    ├── Gold/
    ├── EuroStoxx/
    ├── Bunds/
    ├── GBPUSD/
    ├── JPY/
    └── HeatingOil/
```

Each asset folder should contain one or more futures contract CSV files. If an `AIAgent*.csv` file is present, the pipeline joins it to the OHLC bars and restricts the sample to the common AI/OHLC time window before the train/test split.

---

## Supported Assets

The asset registry automatically discovers subfolders under `data_dir`.

| Asset | Example contracts | Tick size |
|---|---|---:|
| Nasdaq | NQH20, NQM20, NQU20 | 0.25 |
| Gold | GCG24, GCJ24, GCM24, GCQ24 | 0.10 |
| EuroStoxx | VGH22, ... | 0.50 |
| German Bunds | GBL... | 0.01 |
| GBP/USD | 6BH..., ... | 0.01 |
| Japanese Yen | 6JH..., ... | 0.005 |
| Heating Oil | HOF..., ... | 0.01 |

Tick sizes are pinned in `Config.tick_override`. To add a new asset, create a new subfolder under `data_dir`, add its contract CSV files, add a tick-size override if needed, and rerun the notebook.

---

## Setup

Install the required Python packages:

```bash
pip install numpy pandas matplotlib seaborn jupyter
```

Python 3.8 or higher is recommended.

Run the notebook:

```bash
jupyter notebook order_execution_v8.ipynb
```

Then run all cells from top to bottom. The notebook imports the reusable pipeline from `order_execution.py` and the plotting functions from `order_execution_plots.py`.

---

## Configuration

The main parameters are stored in the `Config` dataclass in `order_execution.py`. The notebook sets the primary asset and creates a configuration object:

```python
import order_execution as oe
import order_execution_plots as oe_plots

PRIMARY_ASSET = "Nasdaq"

cfg = oe.Config(
    data_dir="../data/",              # change to "data/" if data is inside repo root
    tau=5,                            # holding period in minutes
    half_life=20,                     # EWMA half-life in tau-bars
    m_states=3,                       # volume states
    n_states=3,                       # volatility states
    k_states=3,                       # direction states
    burn_in=500,
    test_start_frac=0.60,
    fill_prob_threshold=0.60,
    queue_haircut=0.92,
    n_mc_paths=10_000,
    mc_objective="expected_ticks",    # baseline used in Sections 1-2
    use_ai_agent_signal=True,
    restrict_to_ai_overlap=True,
)

registry = cfg.discover()
```

To change the primary asset, modify:

```python
PRIMARY_ASSET = "Gold"
```

and rerun the notebook.

---

## Pipeline

The pipeline is designed to avoid look-ahead bias:

```text
load_data
  -> build_ohlc
  -> compute_ranges
  -> compute_ewma_states
  -> generate_signals
  -> restrict_to_ai_overlap
  -> build_epdf
  -> build_mc_frontier_cache
  -> run_backtest
  -> compute_summary_metrics
```

Usage:

```python
# Single asset
res = oe.run_pipeline(PRIMARY_ASSET, cfg, registry)

# Cross-asset comparison
table = oe.cross_asset_table(list(registry), cfg, registry, objective="expected_ticks")
print(table)
```

`run_pipeline()` returns a dictionary containing the OHLC data, empirical distribution store, Monte Carlo cache, trades, summary metrics, optimal levels, and regime-level diagnostics.

---

## Methodology

### 1. Non-overlapping OHLC bars

The raw one-minute data is resampled into non-overlapping `tau`-minute bars. Bars with insufficient one-minute observations are dropped using the `min_completeness` threshold.

This avoids artificially increasing the sample size with overlapping windows, which would make adjacent observations highly correlated.

### 2. Tick-based range variables

For each bar, the model computes:

| Variable | Definition | Execution interpretation |
|---|---|---|
| `R` | `(high - low) / tick_size` | total intra-bar range |
| `R_up` | `(high - open) / tick_size` | favorable move for sell limits |
| `R_dn` | `(open - low) / tick_size` | favorable move for buy limits |

All ranges are rounded to integer tick units.

### 3. State-conditioned empirical distributions

Each bar is assigned to a regime state:

```text
state = (volume_state, volatility_state, direction_state)
```

The state variables are computed using lagged information only. At bar `j`, the EWMA statistics and regime thresholds use data from bars `0` through `j-1`, not the current or future bar.

The empirical probability distributions are trained only on the in-sample window:

```text
[burn_in, int(n * test_start_frac))
```

Out-of-sample bars are used only for backtesting.

### 4. Monte Carlo execution frontier

For each state and trade direction, the model simulates favorable price movement from the empirical conditional distribution and estimates:

```text
P(fill | limit level ell)
```

A queue haircut is applied to make fill probabilities more conservative:

```text
adjusted_fill_probability = queue_haircut * P(fill | ell)
```

The output is an execution frontier: fill probability as a function of limit level.

---

## Placement Objectives

The project compares three ways to choose the optimal limit level `ell*`.

| Objective | Rule | Interpretation |
|---|---|---|
| `expected_ticks` | maximize `ell * queue_haircut * P(fill | ell)` subject to fill probability clearing the threshold | Aggressive; missed fills are treated as costless |
| `cond_on_fill` | choose the widest feasible `ell` subject to the fill threshold | Also aggressive; focuses only on conditional fill quality |
| `net_of_miss` | maximize expected realized ticks after charging missed fills | Conservative; routes to market when no positive-net limit level is found |

The `expected_ticks` objective is useful as a baseline because it produces clear execution frontiers and many limit orders. The `net_of_miss` objective is more realistic because it penalizes missed-fill outcomes.

---

## Metrics

Two headline metrics are reported separately:

| Metric | Formula | Meaning |
|---|---|---|
| `fill_rate` | filled limit orders / placed limit orders | how often the limit order executes |
| `mean_improvement | filled` | average ticks saved on filled limits | conditional execution quality |
| `gross_edge` | `fill_rate * mean_improvement | filled` | fill-quality metric ignoring missed-fill costs |
| `net_vs_market` | average realized ticks over all limit orders | realized improvement vs crossing at the open, including misses |

The most economically honest metric is `net_vs_market`, because it includes the cost of missed limit orders.

---

## Results

### Cross-asset baseline results

The table below uses `tau = 5`, `theta = 0.60`, and the `expected_ticks` objective.

| Asset | Tick | Limit trades | Fill rate | Impr\|filled | Gross edge | Net vs market |
|---|---:|---:|---:|---:|---:|---:|
| Bunds | 0.010 | 8 | 50.00% | 1.00 | 0.50 | -0.38 |
| EuroStoxx | 0.500 | 48 | 52.08% | 1.00 | 0.52 | -3.65 |
| GBPUSD | 0.010 | 62 | 70.97% | 1.00 | 0.71 | -0.56 |
| Gold | 0.100 | 559 | 67.44% | 1.01 | 0.68 | -4.38 |
| HeatingOil | 0.010 | 117 | 55.56% | 14.83 | 8.24 | -14.19 |
| JPY | 0.005 | 163 | 65.64% | 1.00 | 0.66 | -1.34 |
| Nasdaq | 0.250 | 864 | 64.93% | 3.14 | 2.04 | -10.14 |

### Nasdaq primary-asset result

For Nasdaq, the AI/OHLC overlap correction keeps all 918 out-of-sample signal events in the January-May 2020 AI signal window. Under the baseline objective:

- 918 total out-of-sample signal events
- 864 limit orders placed
- 54 market fallbacks
- 64.93% limit fill rate
- +3.14 ticks average improvement conditional on fill
- +2.04 ticks gross edge
- -10.14 ticks net vs market after missed-fill costs

The result highlights the central execution tradeoff: wider limits improve filled trades, but unfilled orders can lose more than the filled-order gains.

---

## Model Limitations

| Limitation | Impact | Current treatment |
|---|---|---|
| Queue position not modeled directly | Fill probability can be overstated at touched prices | Queue haircut applied |
| No partial fills | Every order is treated as fully filled or fully missed | Simplifying assumption |
| No latency model | Assumes immediate placement at bar open | Simplifying assumption |
| OHLC-only path | Cannot observe the exact intrabar path or queue dynamics | Uses high/low touch condition |
| No explicit adverse selection model | Filled limits may be selected in unfavorable states | Partially reflected through realized net metrics |
| Independent MC sampling | Monte Carlo draws treat bars as conditionally independent | Regime conditioning partially mitigates volatility clustering |
| Thin assets | Some assets have too few out-of-sample events for strong inference | Primary analysis focuses on Nasdaq |

---

## Notebook Sections

| Section | Content |
|---|---|
| 0 | Configuration and asset discovery |
| 1 | Primary-asset pipeline run and execution summary |
| 2 | Execution frontiers, Figure-2 replication, and cumulative improvement plots |
| 3 | Objective comparison: `expected_ticks`, `cond_on_fill`, and `net_of_miss` |
| 4 | Sensitivity analysis over `tau` and fill threshold |
| 5 | Cross-asset comparison table |

---

## References

- Hirsa, A. — IEOR4703 Term Project 2 specification: *Volatility-Volume-based Order Management*
- Almgren, R. and Chriss, N. (2001). *Optimal Execution of Portfolio Transactions*
- Harris, L. (2003). *Trading and Exchanges: Market Microstructure for Practitioners*
