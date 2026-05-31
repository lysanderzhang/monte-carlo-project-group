"""
order_execution.py
==================

State-conditioned limit-order execution engine for the Order-Execution term
project.  Encapsulates the full pipeline so it can be applied to any asset in a
single call:

    from order_execution import Config, OrderExecution, run_pipeline

    cfg = Config(data_dir="../data/")
    oe  = OrderExecution(cfg).run("Nasdaq")     # full run + cached artifacts
    oe.plot_all()                               # frontiers, Figure-2, P&L, calib
    print(oe.summary)

    # loop every asset for the cross-asset table:
    results, cross = run_all(cfg.discover(), cfg)

Pipeline order (corrected, no-leakage):
    load_data -> build_ohlc -> compute_ranges -> compute_ewma_states
    -> generate_signals -> restrict_to_ai_overlap -> build_epdf (in-sample only)
    -> build_mc_frontier_cache -> run_backtest -> compute_summary_metrics

MC objectives (pick the limit level l* per state/direction):
    'expected_ticks' : argmax_l  l * haircut * P(fill | l)      [misses costless]
    'cond_on_fill'   : argmax_l  l   s.t. haircut*P(fill) >= thr [widest feasible]
    'net_of_miss'    : argmax_l  E[realized ticks]              [charges misses]
                       = P(touch)*(q*l + (1-q)*c_touch) + (1-P(touch))*c_miss
                       estimated from the in-sample joint of (extreme, close).
                       l*=0 (market order) whenever no level beats it in-sample.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from functools import reduce
from itertools import product
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# CME/Eurex futures month codes -> calendar month (for ordering contracts).
MONTH_CODE = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
              "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    """All tunable parameters for one run.  Override per call with replace()."""

    # data
    data_dir: str = "../data/"
    min_bars_per_day: int = 350
    min_completeness: float = 0.60
    tick_override: Dict[str, float] = field(default_factory=lambda: {
        "Bunds": 0.01, "EuroStoxx": 0.5, "GBPUSD": 0.01, "Gold": 0.10,
        "HeatingOil": 0.01, "JPY": 0.005, "Nasdaq": 0.25,
    })

    # holding period
    tau: int = 5

    # signal
    use_ai_agent_signal: bool = True
    restrict_to_ai_overlap: bool = True
    signal_type: str = "momentum"        # synthetic fallback only
    signal_lookback: int = 10
    signal_threshold: float = 1.0

    # EWMA state engine
    half_life: int = 20
    m_states: int = 3                    # volume bins
    n_states: int = 3                    # volatility bins
    k_states: int = 3                    # direction bins
    dir_threshold: float = 1.0           # ticks; dead-band for direction state
    vol_proxy: str = "ewmv"              # 'ewmv' | 'range_mean'
    burn_in: int = 500

    # ePDF
    min_epdf_samples: int = 30
    smoothing_alpha: float = 0.5

    # train / test
    test_start_frac: float = 0.60

    # Monte Carlo / objective
    n_mc_paths: int = 10_000
    mc_confidence: float = 0.95
    bootstrap_reps: int = 1_000
    mc_objective: str = "net_of_miss"    # 'expected_ticks'|'cond_on_fill'|'net_of_miss'

    # placement
    fill_prob_threshold: float = 0.60
    queue_haircut: float = 0.92
    min_fill_obs: int = 10               # net_of_miss: min in-sample fills at l*
                                         # (guards against overfitting deep tails)

    random_seed: int = 42

    def tick_for(self, asset: str, prices: Optional[np.ndarray] = None) -> float:
        """Pinned tick if known, else inferred from the price grid."""
        if asset in self.tick_override:
            return self.tick_override[asset]
        if prices is None:
            raise ValueError(f"No pinned tick for {asset} and no prices to infer from.")
        return infer_tick_size(prices)

    def discover(self) -> "AssetRegistry":
        return AssetRegistry(self.data_dir)


# ─────────────────────────────────────────────────────────────────────────────
# DATA DISCOVERY & LOADING
# ─────────────────────────────────────────────────────────────────────────────
def _expiry_key(stem: str) -> Tuple[int, int]:
    """'VGH22' -> (2022, 3); used to order contracts by expiry."""
    try:
        return (2000 + int(stem[-2:]), MONTH_CODE.get(stem[-3].upper(), 0))
    except (ValueError, IndexError):
        return (9999, 0)


def infer_tick_size(prices: np.ndarray) -> float:
    """Price-grid spacing as the GCD of gaps between sorted unique prices."""
    u = np.unique(np.round(np.asarray(prices, dtype=float), 6))
    if len(u) < 2:
        return 1.0
    scale = 10 ** 6
    ints = np.round((u - u.min()) * scale).astype(np.int64)
    ints = ints[ints > 0]
    return reduce(gcd, ints.tolist()) / scale if len(ints) else 1.0


def _excel_to_date(serial: int) -> datetime:
    """Excel serial integer -> datetime (Excel epoch is 1899-12-30)."""
    return datetime(1899, 12, 30) + timedelta(days=int(serial))


class AssetRegistry(dict):
    """Maps asset folder name -> {contracts, ai_agent}. Acts like a dict."""

    def __init__(self, data_dir: str):
        super().__init__()
        base = Path(data_dir)
        if not base.exists():
            return
        for sub in sorted(p for p in base.iterdir() if p.is_dir()):
            csvs = list(sub.glob("*.csv"))
            if not csvs:
                continue
            ai = [c for c in csvs if c.name.lower().startswith("aiagent")]
            contracts = sorted((c for c in csvs if c not in ai),
                               key=lambda p: _expiry_key(p.stem))
            if contracts:
                self[sub.name] = {"folder": sub.name,
                                  "contracts": [c.name for c in contracts],
                                  "ai_agent": ai[0].name if ai else None}

    def has_ai_agent(self, asset: str) -> bool:
        return bool(self.get(asset, {}).get("ai_agent"))


def _parse_contract(filepath: str) -> pd.DataFrame:
    """Parse one contract CSV -> DataFrame[price, volume] indexed by datetime.
    Columns: 0=excel_date 1=hour 2=min 3..6=price 7=volume."""
    rows = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            try:
                ds, hh, mm = int(parts[0]), int(parts[1]), int(parts[2])
                price, vol = float(parts[3]), int(parts[7])
            except (ValueError, IndexError):
                continue
            rows.append({"datetime": _excel_to_date(ds).replace(hour=hh, minute=mm),
                         "price": price, "volume": vol})
    df = pd.DataFrame(rows).set_index("datetime").sort_index()
    return df[df["volume"] > 0]


def _detect_rollover(front: pd.DataFrame, back: pd.DataFrame,
                     min_bars_per_day: int) -> Optional[datetime]:
    """First day where BOTH contracts are liquid and back volume > front volume.
    Requiring liquidity on both sides avoids false early rolls from stale prints."""
    f = pd.DataFrame({"front_vol": front["volume"].resample("D").sum(),
                      "front_n": front["volume"].resample("D").count()})
    b = pd.DataFrame({"back_vol": back["volume"].resample("D").sum(),
                      "back_n": back["volume"].resample("D").count()})
    both = pd.concat([f, b], axis=1).fillna(0)
    cand = both[(both["front_n"] >= min_bars_per_day) &
                (both["back_n"] >= min_bars_per_day) &
                (both["back_vol"] > both["front_vol"])]
    return None if cand.empty else cand.index[0].to_pydatetime()


def _stitch_contracts(dfs: List[pd.DataFrame], min_bars_per_day: int) -> pd.DataFrame:
    if len(dfs) == 1:
        return dfs[0]
    result = dfs[0].copy()
    for back in dfs[1:]:
        roll = _detect_rollover(result, back, min_bars_per_day) or result.index[-1]
        result = pd.concat([result[result.index < roll],
                            back[back.index >= roll]]).sort_index()
    return result


def _filter_active_days(df: pd.DataFrame, min_bars: int) -> pd.DataFrame:
    counts = df.groupby(df.index.date).size()
    valid = counts[counts >= min_bars].index
    return df[pd.Series(df.index.date, index=df.index).isin(valid)]


def load_data(asset: str, registry: AssetRegistry, cfg: Config,
              verbose: bool = False) -> pd.DataFrame:
    """Stitch + clean all contracts for an asset into a 1-min price/volume frame."""
    meta = registry[asset]
    folder = Path(cfg.data_dir) / meta["folder"]
    dfs = []
    for fname in meta["contracts"]:
        fp = folder / fname
        if not fp.exists():
            continue
        df = _parse_contract(str(fp))
        if not df.empty:
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No data for {asset} in {folder}")
    clean = _filter_active_days(_stitch_contracts(dfs, cfg.min_bars_per_day),
                                cfg.min_bars_per_day)
    if verbose:
        print(f"  {asset}: {len(clean):,} bars / "
              f"{clean.index.normalize().nunique()} active days "
              f"({clean.index[0].date()} -> {clean.index[-1].date()})")
    return clean


# ─────────────────────────────────────────────────────────────────────────────
# OHLC, RANGES, STATES
# ─────────────────────────────────────────────────────────────────────────────
def build_ohlc(df: pd.DataFrame, tau: int, min_completeness: float) -> pd.DataFrame:
    """Aggregate 1-min rows into non-overlapping tau-min OHLC bars on TIME
    boundaries.  Windows aligned to the session start; under-complete windows
    (early closes, gaps) are dropped rather than producing a thin bar."""
    out, need = [], max(1, int(np.ceil(min_completeness * tau)))
    for _, day in df.groupby(df.index.normalize()):
        agg = day.resample(f"{tau}min", origin="start_day").agg(
            open=("price", "first"), high=("price", "max"),
            low=("price", "min"), close=("price", "last"),
            volume=("volume", "sum"), n_min=("price", "size"))
        out.append(agg[agg["n_min"] >= need])
    ohlc = pd.concat(out).dropna(subset=["open", "high", "low", "close"])
    return ohlc.drop(columns="n_min").sort_index()


def compute_ranges(ohlc: pd.DataFrame, tick_size: float) -> pd.DataFrame:
    """Tick-quantised ranges plus the signed open-to-open and intrabar moves."""
    df = ohlc.copy()
    df["R"] = ((df["high"] - df["low"]) / tick_size).round().astype(int).clip(lower=0)
    df["R_up"] = ((df["high"] - df["open"]) / tick_size).round().astype(int).clip(lower=0)
    df["R_dn"] = ((df["open"] - df["low"]) / tick_size).round().astype(int).clip(lower=0)
    df["delta_x"] = df["open"].diff()                       # open_j - open_{j-1}
    df["delta_x_intrabar"] = df["close"] - df["open"]       # synthetic signal only
    return df


def _ewma_recursive(series: np.ndarray, half_life: int) -> Tuple[np.ndarray, np.ndarray]:
    """Recursive EWM mean and std (Algorithm 1).  Uses series[j-1] at step j
    (strict lag-1) so nothing is forward-looking."""
    lam = 2 ** (-1.0 / half_life)
    n = len(series)
    ewma, ewmstd = np.zeros(n), np.zeros(n)
    sumW = sumWX = sumWSS = 0.0
    for j in range(1, n):
        eta = series[j - 1]
        if j == 1:
            sumW, sumWX, ewma[j] = 1.0, eta, eta
        else:
            sumW = lam * sumW + 1.0
            sumWX = lam * sumWX + eta
            ewma[j] = sumWX / sumW
            sumWSS = lam * sumWSS + (eta - ewma[j]) ** 2
            ewmstd[j] = np.sqrt(sumWSS / sumW)
    return ewma, ewmstd


def _bin_to_states(value: float, history: np.ndarray, n_states: int) -> int:
    """Quantile bin of `value` within `history` (strictly prior observations)."""
    if len(history) < n_states + 1:
        return 0
    thr = np.percentile(history, np.linspace(0, 100, n_states + 1)[1:-1])
    return int(np.searchsorted(thr, value, side="right"))


def compute_ewma_states(df: pd.DataFrame, tick_size: float, cfg: Config) -> pd.DataFrame:
    """Add vol_state / sig_state / dir_state.  All states are strictly
    out-of-sample; bars before burn_in get state = -1."""
    df = df.copy()
    n = len(df)
    ewma_volume, _ = _ewma_recursive(df["volume"].values.astype(float), cfg.half_life)
    ewma_rng, ewmstd_rng = _ewma_recursive(df["R"].values.astype(float), cfg.half_life)
    vol_signal = ewmstd_rng if cfg.vol_proxy == "ewmv" else ewma_rng
    dx = df["delta_x"].values

    vol_state = np.full(n, -1, int)
    sig_state = np.full(n, -1, int)
    dir_state = np.full(n, -1, int)
    dir_tick = cfg.dir_threshold * tick_size

    for j in range(cfg.burn_in, n):
        vol_state[j] = _bin_to_states(ewma_volume[j], ewma_volume[1:j], cfg.m_states)
        sig_state[j] = _bin_to_states(vol_signal[j], vol_signal[1:j], cfg.n_states)
        pdx = dx[j - 1] if not np.isnan(dx[j - 1]) else 0.0
        dir_state[j] = 0 if pdx < -dir_tick else (2 if pdx > dir_tick else 1)

    df["vol_state"], df["sig_state"], df["dir_state"] = vol_state, sig_state, dir_state
    df["ewma_volume"], df["vol_signal"] = ewma_volume, vol_signal
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SIGNALS
# ─────────────────────────────────────────────────────────────────────────────
def load_ai_agent_signal(asset: str, registry: AssetRegistry, cfg: Config) -> pd.DataFrame:
    meta = registry[asset]
    fpath = Path(cfg.data_dir) / meta["folder"] / meta["ai_agent"]
    rows = []
    with open(fpath) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            try:
                dt = _excel_to_date(int(parts[0])).replace(hour=int(parts[1]),
                                                            minute=int(parts[2]))
                price, position = float(parts[3]), int(parts[4])
            except (ValueError, IndexError):
                continue
            rows.append({"datetime": dt, "ai_price": price, "position": position})
    ai = pd.DataFrame(rows).set_index("datetime").sort_index()
    ai["order"] = ai["position"].diff().fillna(0).astype(int)
    ai["signal"] = np.sign(ai["order"]).astype(int)
    ai["order_size"] = ai["order"].abs()
    return ai


def generate_signals(df: pd.DataFrame, asset: str, tick_size: float,
                     registry: AssetRegistry, cfg: Config) -> pd.DataFrame:
    """Attach a +1/0/-1 trade signal: AI-agent CSV if available, else synthetic."""
    df = df.copy()
    if cfg.use_ai_agent_signal and registry.has_ai_agent(asset):
        ai = load_ai_agent_signal(asset, registry, cfg)
        df = df.join(ai[["position", "order", "signal", "order_size"]], how="left")
        for c, fill in [("signal", 0), ("order_size", 0), ("position", 0)]:
            df[c] = df[c].fillna(fill).astype(int)
    else:
        thr = cfg.signal_threshold * tick_size
        if cfg.signal_type == "momentum":
            raw = df["delta_x_intrabar"].ewm(halflife=cfg.signal_lookback,
                                             adjust=False).mean().shift(1)
        elif cfg.signal_type == "mean_reversion":
            raw = -(df["close"] - df["close"].rolling(cfg.signal_lookback).mean()).shift(1)
        else:
            raise ValueError(cfg.signal_type)
        df["signal"] = np.where(raw > thr, 1, np.where(raw < -thr, -1, 0)).astype(int)
        df["order_size"], df["position"] = 1, 0
    return df


def restrict_to_ai_overlap(df: pd.DataFrame, asset: str,
                           registry: AssetRegistry, cfg: Config) -> pd.DataFrame:
    """Clip OHLC+signal to the window where BOTH OHLC and AI signal exist.
    Call after generate_signals() and before build_epdf()."""
    if not (cfg.use_ai_agent_signal and cfg.restrict_to_ai_overlap
            and registry.has_ai_agent(asset)):
        return df
    ai = load_ai_agent_signal(asset, registry, cfg)
    start, end = max(df.index.min(), ai.index.min()), min(df.index.max(), ai.index.max())
    if start > end:
        raise ValueError(f"No AI/OHLC overlap for {asset}.")
    return df.loc[start:end].copy()


# ─────────────────────────────────────────────────────────────────────────────
# ePDF STORE  (conditional empirical PMFs + joint samples for net_of_miss)
# ─────────────────────────────────────────────────────────────────────────────
class EpdfStore:
    """Online conditional empirical PMFs for R, R_up, R_dn keyed by state, plus
    the in-sample joint of (favourable extreme, signed close move) needed for the
    miss-aware objective."""

    def __init__(self, smoothing_alpha: float, min_samples: int):
        self.alpha = smoothing_alpha
        self.min_samples = min_samples
        self._counts = defaultdict(
            lambda: {"R": np.zeros(1), "R_up": np.zeros(1), "R_dn": np.zeros(1)})
        self._marginal = {"R": np.zeros(1), "R_up": np.zeros(1), "R_dn": np.zeros(1)}
        self._samples = defaultdict(lambda: {"R_up": [], "R_dn": [], "dx_close": []})
        self._marginal_samples = {"R_up": [], "R_dn": [], "dx_close": []}

    @staticmethod
    def _extend(arr, size):
        return np.append(arr, np.zeros(size - len(arr))) if len(arr) < size else arr

    def update(self, state, R, R_up, R_dn, dx_close):
        for store in (self._counts[state], self._marginal):
            store["R"] = self._extend(store["R"], R + 1)
            store["R_up"] = self._extend(store["R_up"], R_up + 1)
            store["R_dn"] = self._extend(store["R_dn"], R_dn + 1)
            store["R"][R] += 1
            store["R_up"][R_up] += 1
            store["R_dn"][R_dn] += 1
        for s in (self._samples[state], self._marginal_samples):
            s["R_up"].append(R_up)
            s["R_dn"].append(R_dn)
            s["dx_close"].append(dx_close)

    def _pmf(self, counts):
        s = counts + self.alpha
        return s / s.sum()

    def _survival(self, counts):
        pmf = self._pmf(counts)
        return 1.0 - np.cumsum(pmf) + pmf                # P(X >= l)

    def _tot(self, state):
        return int(self._counts[state]["R_up"].sum())

    def _marg(self):
        return int(self._marginal["R_up"].sum())

    def get_pmf(self, state, d):
        if self._tot(state) >= self.min_samples:
            return self._pmf(self._counts[state][d])
        if self._marg() >= self.min_samples:
            return self._pmf(self._marginal[d])
        return np.array([1.0])                           # tier 3 -> market order

    def get_survival(self, state, d):
        if self._tot(state) >= self.min_samples:
            return self._survival(self._counts[state][d])
        if self._marg() >= self.min_samples:
            return self._survival(self._marginal[d])
        return np.array([1.0])

    def get_joint(self, state, d):
        """Return (extreme_samples, miss_improvement_samples) for direction d,
        state cell if it has enough data else the marginal, else None (market)."""
        if self._tot(state) >= self.min_samples:
            s = self._samples[state]
        elif self._marg() >= self.min_samples:
            s = self._marginal_samples
        else:
            return None
        ext = np.asarray(s[d], dtype=float)
        dxc = np.asarray(s["dx_close"], dtype=float)
        miss = dxc if d == "R_up" else -dxc              # realised ticks on a miss
        return ext, miss

    def fill_prob(self, state, d, ell):
        sf = self.get_survival(state, d)
        return 0.0 if ell >= len(sf) else float(sf[ell])

    def state_count(self, state):
        return self._tot(state)


def build_epdf(df: pd.DataFrame, tick_size: float, cfg: Config) -> Tuple[EpdfStore, int]:
    """Train the store ONLY on in-sample bars [burn_in, test_start_idx).  Test
    bars never enter it, so every optimal level is strictly out-of-sample."""
    store = EpdfStore(cfg.smoothing_alpha, cfg.min_epdf_samples)
    n = len(df)
    test_start_idx = int(n * cfg.test_start_frac)
    vs, ss, ds = df["vol_state"].values, df["sig_state"].values, df["dir_state"].values
    R, Ru, Rd = df["R"].values, df["R_up"].values, df["R_dn"].values
    o, c = df["open"].values, df["close"].values

    for j in range(cfg.burn_in, test_start_idx):
        if vs[j] < 0 or ss[j] < 0 or ds[j] < 0:
            continue
        dx_close = (c[j] - o[j]) / tick_size             # signed, continuous ticks
        store.update((vs[j], ss[j], ds[j]), int(R[j]), int(Ru[j]), int(Rd[j]), dx_close)
    return store, test_start_idx


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO EXECUTION FRONTIER  (+ net_of_miss objective)
# ─────────────────────────────────────────────────────────────────────────────
def mc_simulate_execution(store: EpdfStore, state, direction, cfg: Config,
                          fill_prob_threshold=None, objective=None,
                          bootstrap_reps=None) -> Dict:
    """Monte Carlo over candidate limit levels for one (state, direction).

    The fill-probability frontier (fill_rate, CI band) is objective-independent.
    The chosen optimal_ell depends on `objective`:
      expected_ticks : argmax l*haircut*P(fill), feasible: haircut*P(fill)>=thr
      cond_on_fill   : argmax l                 , feasible: haircut*P(fill)>=thr
      net_of_miss    : argmax E[realised ticks] over l in {0,1,...}; charges
                       missed fills at the realised open->close move; l*=0 is the
                       market-order fallback chosen whenever no level beats it.
    """
    thr = cfg.fill_prob_threshold if fill_prob_threshold is None else fill_prob_threshold
    obj = cfg.mc_objective if objective is None else objective
    reps = cfg.bootstrap_reps if bootstrap_reps is None else bootstrap_reps
    q = cfg.queue_haircut
    rng = np.random.default_rng(cfg.random_seed)

    pmf = store.get_pmf(state, direction)
    L = len(pmf)
    samples = rng.choice(L, size=cfg.n_mc_paths, p=pmf)
    ell_grid = np.arange(1, max(2, L))
    fill_rates = q * np.array([(samples >= e).mean() for e in ell_grid])
    exp_improvement = ell_grid * fill_rates              # expected_ticks objective

    # bootstrap CI on the fill rate (objective-independent frontier band)
    counts = store._counts[state][direction]
    obs = np.repeat(np.arange(len(counts)), counts.astype(int))
    ci_lo, ci_hi = np.zeros(len(ell_grid)), np.ones(len(ell_grid))
    if reps > 0 and len(obs) >= store.min_samples:
        idx = rng.integers(0, len(obs), size=(reps, len(obs)))
        bs = obs[idx]
        boot = np.empty((reps, len(ell_grid)))
        for i, e in enumerate(ell_grid):
            boot[:, i] = (bs >= e).mean(axis=1)
        a = (1 - cfg.mc_confidence) / 2
        ci_lo = q * np.percentile(boot, 100 * a, axis=0)
        ci_hi = q * np.percentile(boot, 100 * (1 - a), axis=0)

    exp_net = np.full(len(ell_grid), np.nan)
    optimal_ell, is_market = 0, True
    feasible = np.where(fill_rates >= thr)[0]            # haircut*P(fill) >= threshold

    if obj == "net_of_miss":
        # Among feasible levels (so execution likelihood is preserved and the
        # conditional estimates are well-sampled), pick the one that maximises
        # expected REALISED ticks, charging missed fills at the open->close move.
        joint = store.get_joint(state, direction)
        if joint is not None and len(feasible):
            ext, miss = joint
            exp_net = np.full(len(ell_grid), -np.inf)
            for i in feasible:
                e = ell_grid[i]
                touch = ext >= e
                if touch.sum() < cfg.min_fill_obs:        # too few in-sample fills
                    continue
                pt = touch.mean()
                c_touch = miss[touch].mean()                            # queue miss
                c_miss = miss[~touch].mean() if (~touch).any() else 0.0  # geom. miss
                exp_net[i] = pt * (q * e + (1 - q) * c_touch) + (1 - pt) * c_miss
            best = int(np.argmax(exp_net))
            if np.isfinite(exp_net[best]) and exp_net[best] > 0.0:  # else market order
                optimal_ell, is_market = int(ell_grid[best]), False
    elif len(feasible):
        if obj == "cond_on_fill":
            best = feasible[np.argmax(ell_grid[feasible])]
        else:                                             # expected_ticks
            best = feasible[np.argmax(exp_improvement[feasible])]
        optimal_ell, is_market = int(ell_grid[best]), False

    return dict(ell_grid=ell_grid, fill_rate=fill_rates,
                exp_improvement=exp_improvement, exp_net=exp_net,
                ci_fill_lo=ci_lo, ci_fill_hi=ci_hi,
                optimal_ell=optimal_ell, is_market_order=is_market, objective=obj)


def build_mc_frontier_cache(store: EpdfStore, cfg: Config, fill_prob_threshold=None,
                            objective=None, bootstrap_reps=None) -> Dict:
    """Pre-compute MC results for every (state, direction) so the backtest is O(1)/bar."""
    cache = {}
    for vs, ss, ds in product(range(cfg.m_states), range(cfg.n_states), range(cfg.k_states)):
        st = (vs, ss, ds)
        for d in ("R_up", "R_dn"):
            cache[(st, d)] = mc_simulate_execution(
                store, st, d, cfg, fill_prob_threshold=fill_prob_threshold,
                objective=objective, bootstrap_reps=bootstrap_reps)
    return cache


def optimal_levels_table(cache: Dict, cfg: Config) -> pd.DataFrame:
    """Tidy l* table across all 27 states for both directions."""
    rows = []
    for vs, ss, ds in product(range(cfg.m_states), range(cfg.n_states), range(cfg.k_states)):
        st = (vs, ss, ds)
        ru, rd = cache[(st, "R_up")], cache[(st, "R_dn")]
        rows.append({"state": st, "vol": vs, "vol_sig": ss, "dir": ds,
                     "sell_ell": ru["optimal_ell"], "buy_ell": rd["optimal_ell"],
                     "sell_market": ru["is_market_order"], "buy_market": rd["is_market_order"]})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST & METRICS
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, mc_cache: Dict, test_start_idx: int,
                 tick_size: float) -> pd.DataFrame:
    """Replay OOS bars: place the state's optimal limit; fill if the bar's
    favourable extreme reaches it, else fall back to the bar close."""
    records = []
    for j in range(test_start_idx, len(df)):
        row = df.iloc[j]
        if row["signal"] == 0:
            continue
        vs, ss, ds = int(row["vol_state"]), int(row["sig_state"]), int(row["dir_state"])
        if vs < 0 or ss < 0 or ds < 0:
            continue
        state, sgn = (vs, ss, ds), int(row["signal"])
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        direction = "R_up" if sgn == -1 else "R_dn"      # sell->R_up, buy->R_dn
        mc = mc_cache.get((state, direction))
        ell = 0 if mc is None else mc["optimal_ell"]
        is_mkt = True if mc is None else mc["is_market_order"]

        if is_mkt or ell == 0:
            limit = fill = o
            filled, missed, impr, is_mkt = True, False, 0.0, True
        else:
            if sgn == -1:
                limit = o + ell * tick_size
                filled = h >= limit
            else:
                limit = o - ell * tick_size
                filled = l <= limit
            if filled:
                fill, missed = limit, False
                impr = (fill - o) / tick_size if sgn == -1 else (o - fill) / tick_size
            else:
                fill, missed = c, True                   # missed -> execute at close
                impr = (c - o) / tick_size if sgn == -1 else (o - c) / tick_size

        records.append(dict(bar_time=df.index[j], signal=sgn, vol_state=vs,
                            sig_state=ss, dir_state=ds, state=state, direction=direction,
                            open_price=o, limit_level_ticks=ell, limit_price=limit,
                            filled=filled, fill_price=fill, improvement_ticks=impr,
                            is_market_fallback=is_mkt, is_missed_limit=missed))
    return pd.DataFrame(records).set_index("bar_time") if records else pd.DataFrame()


def compute_summary_metrics(trades: pd.DataFrame, asset_label: str,
                            cfg: Config) -> pd.Series:
    """Headline execution-quality metrics.  Two are reported because they answer
    different questions:
      gross_edge      = fill_rate * mean improvement | filled   (ignores misses)
      net_vs_market   = mean realised ticks over ALL limit orders, i.e. the true
                        average vs a market order at the open (charges misses)."""
    lim = trades[~trades["is_market_fallback"]]
    filled = lim[lim["filled"]]
    missed = lim[lim["is_missed_limit"]]
    fr = lim["filled"].mean() if len(lim) else np.nan
    mif = filled["improvement_ticks"].mean() if len(filled) else np.nan
    return pd.Series({
        "Asset": asset_label,
        "Total trades": len(trades),
        "Limit orders placed": len(lim),
        "Market fallbacks": len(trades) - len(lim),
        "Limit fill rate": f"{fr:.2%}" if len(lim) else "N/A",
        "Missed fill rate": f"{lim['is_missed_limit'].mean():.2%}" if len(lim) else "N/A",
        "Mean improvement | filled": f"{mif:.2f} ticks",
        "Mean cost | missed fill": (f"{missed['improvement_ticks'].mean():.2f} ticks"
                                    if len(missed) else "N/A"),
        "gross_edge (fills only)": f"{fr * mif:.2f} ticks" if len(filled) else "N/A",
        "net_vs_market (incl. misses)": (f"{lim['improvement_ticks'].mean():.2f} ticks"
                                         if len(lim) else "N/A"),
        "Queue haircut applied": f"{cfg.queue_haircut:.0%}",
    })


def regime_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    """Per (vol_state, sig_state) fill rate / miss rate / realised net (limit orders)."""
    lim = trades[~trades["is_market_fallback"]]
    if lim.empty:
        return pd.DataFrame()
    return (lim.groupby(["vol_state", "sig_state"])
            .agg(n=("filled", "count"), fill_rate=("filled", "mean"),
                 miss=("is_missed_limit", "mean"), net=("improvement_ticks", "mean"))
            .round(3))


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(asset: str, cfg: Config, registry: Optional[AssetRegistry] = None,
                 *, tau=None, fill_threshold=None, objective=None,
                 verbose: bool = False) -> Dict:
    """Full corrected pipeline for one asset.  Returns a dict with every artifact
    (ohlc, store, mc_cache, trades, summary, metrics) plus scalar metrics suitable
    for sweeps and the cross-asset table.  `tau/fill_threshold/objective` override
    the config for this run only."""
    registry = registry or cfg.discover()
    run_cfg = replace(cfg,
                      tau=cfg.tau if tau is None else tau,
                      fill_prob_threshold=(cfg.fill_prob_threshold
                                           if fill_threshold is None else fill_threshold),
                      mc_objective=cfg.mc_objective if objective is None else objective)

    raw = load_data(asset, registry, run_cfg, verbose=verbose)
    tick = run_cfg.tick_for(asset, raw["price"].values)
    ohlc = compute_ranges(build_ohlc(raw, run_cfg.tau, run_cfg.min_completeness), tick)
    ohlc = compute_ewma_states(ohlc, tick, run_cfg)
    ohlc = generate_signals(ohlc, asset, tick, registry, run_cfg)
    ohlc = restrict_to_ai_overlap(ohlc, asset, registry, run_cfg)

    base = dict(asset=asset, tick=tick, objective=run_cfg.mc_objective,
                sample_start=ohlc.index.min() if len(ohlc) else pd.NaT,
                sample_end=ohlc.index.max() if len(ohlc) else pd.NaT)

    if len(ohlc) <= run_cfg.burn_in + 10:
        return {**base, "status": "too_few_rows", "n_trades": 0,
                "fill_rate": np.nan, "net_vs_market": np.nan}

    store, tsi = build_epdf(ohlc, tick, run_cfg)
    cache = build_mc_frontier_cache(store, run_cfg)
    trades = run_backtest(ohlc, cache, tsi, tick)

    if trades.empty:
        return {**base, "status": "no_trades", "n_trades": 0,
                "fill_rate": np.nan, "net_vs_market": np.nan,
                "ohlc": ohlc, "store": store, "mc_cache": cache, "test_start_idx": tsi}

    lim = trades[~trades["is_market_fallback"]]
    filled = lim[lim["filled"]]
    missed = lim[lim["is_missed_limit"]]
    fr = lim["filled"].mean() if len(lim) else np.nan
    mif = filled["improvement_ticks"].mean() if len(filled) else np.nan

    return {**base, "status": "ok",
            "test_start": ohlc.index[tsi],
            "active_signals_total": int((ohlc["signal"] != 0).sum()),
            "active_signals_oos": int((ohlc.iloc[tsi:]["signal"] != 0).sum()),
            "total_trades": len(trades), "limit_orders": len(lim),
            "market_fallbacks": len(trades) - len(lim), "n_trades": len(lim),
            "fill_rate": fr, "mean_improvement": mif,
            "mean_miss_cost": missed["improvement_ticks"].mean() if len(missed) else np.nan,
            "gross_edge": fr * mif if len(filled) else np.nan,
            "net_vs_market": lim["improvement_ticks"].mean() if len(lim) else np.nan,
            # artifacts for plotting / inspection
            "ohlc": ohlc, "store": store, "mc_cache": cache, "test_start_idx": tsi,
            "trades": trades,
            "summary": compute_summary_metrics(trades, asset, run_cfg),
            "optimal_levels": optimal_levels_table(cache, run_cfg),
            "regime": regime_breakdown(trades)}


def cross_asset_table(assets: List[str], cfg: Config,
                      registry: Optional[AssetRegistry] = None,
                      objective=None) -> pd.DataFrame:
    """Run every asset and assemble the comparison table (honest net column)."""
    registry = registry or cfg.discover()
    rows = []
    for a in assets:
        try:
            r = run_pipeline(a, cfg, registry, objective=objective)
        except Exception as exc:                         # keep the table going
            rows.append({"Asset": a, "status": f"error: {exc}"})
            continue
        fr, mif = r.get("fill_rate", np.nan), r.get("mean_improvement", np.nan)
        rows.append({"Asset": a, "Tick": r.get("tick"),
                     "Limit trades": r.get("n_trades", 0),
                     "Fill rate": f"{fr:.2%}" if pd.notna(fr) else "N/A",
                     "Impr|filled": f"{mif:.2f}" if pd.notna(mif) else "N/A",
                     "gross_edge": (f"{r['gross_edge']:.2f}"
                                    if pd.notna(r.get("gross_edge", np.nan)) else "N/A"),
                     "net_vs_market": (f"{r['net_vs_market']:.2f}"
                                       if pd.notna(r.get("net_vs_market", np.nan)) else "N/A")})
    return pd.DataFrame(rows).set_index("Asset")


class OrderExecution:
    """Thin object wrapper around run_pipeline that caches artifacts and exposes
    the plots.  Lets the notebook stay a few lines per asset."""

    def __init__(self, cfg: Config, registry: Optional[AssetRegistry] = None):
        self.cfg = cfg
        self.registry = registry or cfg.discover()
        self.res: Dict = {}

    def run(self, asset: str, **overrides) -> "OrderExecution":
        self.res = run_pipeline(asset, self.cfg, self.registry, **overrides)
        return self

    def __getattr__(self, name):                         # expose res keys as attrs
        res = self.__dict__.get("res", {})
        if name in res:
            return res[name]
        raise AttributeError(name)

    # --- plots (imported lazily so the core module has no matplotlib dependency) ---
    def plot_all(self):
        from oe_plots import plot_execution_frontiers, plot_figure2, \
            plot_cumulative_improvement, plot_calibration
        plot_execution_frontiers(self.res, self.cfg)
        plot_figure2(self.res, self.cfg)
        plot_cumulative_improvement(self.res, self.cfg)
        plot_calibration(self.res, self.cfg)
