"""
pipeline_fixed.py — corrected core logic, used to validate fixes before
they go into the notebook. Not the deliverable; the notebook is.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta, date
from pathlib import Path
from itertools import product
from typing import Dict, List, Optional, Tuple

EXCEL_EPOCH = datetime(1899, 12, 30)


def _excel_to_date(serial: int) -> datetime:
    return EXCEL_EPOCH + timedelta(days=int(serial))


def _parse_contract(filepath: str) -> pd.DataFrame:
    rows = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
            try:
                ds = int(parts[0]); hh = int(parts[1]); mm = int(parts[2])
                price = float(parts[3]); vol = int(parts[7])
            except (ValueError, IndexError):
                continue
            dt = _excel_to_date(ds).replace(hour=hh, minute=mm)
            rows.append({'datetime': dt, 'price': price, 'volume': vol})
    df = pd.DataFrame(rows).set_index('datetime').sort_index()
    return df[df['volume'] > 0]


# ---------------- FIX #4: time-based OHLC windows ----------------
def build_ohlc(df: pd.DataFrame, tau: int, min_completeness: float = 0.6) -> pd.DataFrame:
    """
    Aggregate 1-min rows into NON-OVERLAPPING tau-min OHLC bars using TIME
    boundaries (resample), not row position. A window that is missing too many
    of its tau minutes is dropped, so a '5-min' bar can never silently span 7
    wall-clock minutes because of data gaps.
    """
    out = []
    for d, day_df in df.groupby(df.index.normalize()):
        agg = day_df.resample(f'{tau}min', origin='start_day').agg(
            open=('price', 'first'),
            high=('price', 'max'),
            low=('price', 'min'),
            close=('price', 'last'),
            volume=('volume', 'sum'),
            n_min=('price', 'size'),
        )
        agg = agg[agg['n_min'] >= max(1, int(np.ceil(min_completeness * tau)))]
        out.append(agg)
    ohlc = pd.concat(out).dropna(subset=['open', 'high', 'low', 'close'])
    return ohlc.drop(columns='n_min').sort_index()


def compute_ranges(ohlc: pd.DataFrame, tick_size: float) -> pd.DataFrame:
    df = ohlc.copy()
    df['R'] = ((df['high'] - df['low']) / tick_size).round().astype(int).clip(lower=0)
    df['R_up'] = ((df['high'] - df['open']) / tick_size).round().astype(int).clip(lower=0)
    df['R_dn'] = ((df['open'] - df['low']) / tick_size).round().astype(int).clip(lower=0)
    # FIX #6 (definition): spec defines dx as open-to-open across bars.
    df['delta_x_intrabar'] = df['close'] - df['open']            # kept for signal use
    df['delta_x'] = df['open'].diff()                            # open_j - open_{j-1}
    return df


def _ewma_recursive(series, half_life):
    lam = 2 ** (-1.0 / half_life)
    n = len(series)
    ewma = np.zeros(n); ewmstd = np.zeros(n)
    sumW = sumWX = sumWSS = 0.0
    for j in range(1, n):
        eta = series[j - 1]
        if j == 1:
            sumW, sumWX = 1.0, eta
            ewma[j] = eta; sumWSS = 0.0; ewmstd[j] = 0.0
        else:
            sumW = lam * sumW + 1.0
            sumWX = lam * sumWX + eta
            ewma[j] = sumWX / sumW
            sumWSS = lam * sumWSS + (eta - ewma[j]) ** 2
            ewmstd[j] = np.sqrt(sumWSS / sumW)
    return ewma, ewmstd


def _bin(value, history, n_states):
    if len(history) < n_states + 1:
        return 0
    q = np.linspace(0, 100, n_states + 1)[1:-1]
    thr = np.percentile(history, q)
    return int(np.searchsorted(thr, value, side='right'))


def compute_ewma_states(df, tick_size, half_life, m_states, n_states, k_states,
                        dir_threshold, burn_in, vol_proxy='ewmv'):
    """
    FIX #1: volatility state can use the EWMV (std) output of Algorithm 1
    (vol_proxy='ewmv'), matching the spec, instead of EWMA-mean-of-range.
    FIX #6: direction uses open-to-open dx_{j-1} = open_j - open_{j-1}, lag-1.
    """
    df = df.copy(); n = len(df)
    volumes = df['volume'].values.astype(float)
    ranges = df['R'].values.astype(float)
    dx = df['delta_x'].values   # open-to-open

    ewma_volume, _ = _ewma_recursive(volumes, half_life)
    ewma_rng, ewmstd_rng = _ewma_recursive(ranges, half_life)
    vol_signal = ewmstd_rng if vol_proxy == 'ewmv' else ewma_rng

    vol_state = np.full(n, -1, int)
    sig_state = np.full(n, -1, int)
    dir_state = np.full(n, -1, int)
    dir_tick = dir_threshold * tick_size

    for j in range(burn_in, n):
        vol_state[j] = _bin(ewma_volume[j], ewma_volume[1:j], m_states)
        sig_state[j] = _bin(vol_signal[j], vol_signal[1:j], n_states)
        pdx = dx[j - 1] if not np.isnan(dx[j - 1]) else 0.0
        dir_state[j] = 0 if pdx < -dir_tick else (2 if pdx > dir_tick else 1)

    df['vol_state'] = vol_state
    df['sig_state'] = sig_state
    df['dir_state'] = dir_state
    df['ewma_volume'] = ewma_volume
    df['vol_signal'] = vol_signal
    return df


class EpdfStore:
    def __init__(self, alpha=0.5, min_samples=30):
        self.alpha = alpha; self.min_samples = min_samples
        self._counts = defaultdict(lambda: {'R': np.zeros(1), 'R_up': np.zeros(1), 'R_dn': np.zeros(1)})
        self._marginal = {'R': np.zeros(1), 'R_up': np.zeros(1), 'R_dn': np.zeros(1)}

    @staticmethod
    def _extend(arr, size):
        return np.append(arr, np.zeros(size - len(arr))) if len(arr) < size else arr

    def update(self, state, R, R_up, R_dn):
        for store in (self._counts[state], self._marginal):
            store['R'] = self._extend(store['R'], R + 1)
            store['R_up'] = self._extend(store['R_up'], R_up + 1)
            store['R_dn'] = self._extend(store['R_dn'], R_dn + 1)
            store['R'][R] += 1; store['R_up'][R_up] += 1; store['R_dn'][R_dn] += 1

    def _pmf(self, counts):
        s = counts + self.alpha
        return s / s.sum()

    def _survival(self, counts):
        pmf = self._pmf(counts)
        return 1.0 - np.cumsum(pmf) + pmf

    def _tot(self, state):
        return int(self._counts[state]['R_up'].sum())

    def _marg(self):
        return int(self._marginal['R_up'].sum())

    def get_pmf(self, state, d):
        if self._tot(state) >= self.min_samples:
            return self._pmf(self._counts[state][d])
        if self._marg() >= self.min_samples:
            return self._pmf(self._marginal[d])
        return np.array([1.0])

    def get_survival(self, state, d):
        if self._tot(state) >= self.min_samples:
            return self._survival(self._counts[state][d])
        if self._marg() >= self.min_samples:
            return self._survival(self._marginal[d])
        return np.array([1.0])

    def fill_prob(self, state, d, ell):
        sf = self.get_survival(state, d)
        return 0.0 if ell >= len(sf) else float(sf[ell])

    def state_count(self, state):
        return self._tot(state)


# ---------------- FIX #2: train only on in-sample bars ----------------
def build_epdf(df, burn_in, test_start_frac, alpha, min_samples):
    """
    CRITICAL FIX: the store is updated ONLY with bars in
    [burn_in, test_start_idx).  Test bars never enter the ePDF, so the MC
    cache and every optimal level are strictly out-of-sample.
    """
    store = EpdfStore(alpha=alpha, min_samples=min_samples)
    n = len(df)
    test_start_idx = int(n * test_start_frac)
    vs = df['vol_state'].values; ss = df['sig_state'].values; ds = df['dir_state'].values
    R = df['R'].values; Ru = df['R_up'].values; Rd = df['R_dn'].values
    for j in range(burn_in, test_start_idx):          # <-- stops at test split
        if vs[j] < 0 or ss[j] < 0 or ds[j] < 0:
            continue
        store.update((vs[j], ss[j], ds[j]), int(R[j]), int(Ru[j]), int(Rd[j]))
    return store, test_start_idx


# ---------------- FIX #3: explicit, documented objective ----------------
def mc_simulate_execution(store, state, direction, n_paths, fill_prob_threshold,
                          queue_haircut, objective='expected_ticks', seed=42):
    """
    objective:
      'expected_ticks'    -> maximize ell * P(fill)         (expected ticks captured)
      'cond_on_fill'      -> maximize ell                   (best price | fill), s.t. constraint
    Constraint in BOTH cases: haircut * P(fill) >= threshold.
    """
    rng = np.random.default_rng(seed)
    pmf = store.get_pmf(state, direction)
    L = len(pmf)
    samples = rng.choice(L, size=n_paths, p=pmf)
    ell_grid = np.arange(1, max(2, L))
    fill_rates = queue_haircut * np.array([(samples >= e).mean() for e in ell_grid])
    exp_ticks = ell_grid * fill_rates

    feasible = np.where(fill_rates >= fill_prob_threshold)[0]
    if len(feasible) == 0:
        return dict(ell_grid=ell_grid, fill_rate=fill_rates, exp_improvement=exp_ticks,
                    optimal_ell=0, is_market_order=True, objective=objective)
    if objective == 'cond_on_fill':
        best = feasible[np.argmax(ell_grid[feasible])]      # widest feasible level
    else:
        best = feasible[np.argmax(exp_ticks[feasible])]     # best expected ticks
    return dict(ell_grid=ell_grid, fill_rate=fill_rates, exp_improvement=exp_ticks,
                optimal_ell=int(ell_grid[best]), is_market_order=False, objective=objective)


def build_mc_cache(store, m, n, k, fill_prob_threshold, queue_haircut, objective):
    cache = {}
    for vs, ss, ds in product(range(m), range(n), range(k)):
        st = (vs, ss, ds)
        for d in ('R_up', 'R_dn'):
            cache[(st, d)] = mc_simulate_execution(
                store, st, d, 10000, fill_prob_threshold, queue_haircut, objective)
    return cache


def run_backtest(df, mc_cache, test_start_idx, tick_size):
    rec = []
    vals = df[['signal', 'vol_state', 'sig_state', 'dir_state',
               'open', 'high', 'low', 'close']].values
    idx = df.index
    col = {c: i for i, c in enumerate(['signal', 'vol_state', 'sig_state',
                                       'dir_state', 'open', 'high', 'low', 'close'])}
    for j in range(test_start_idx, len(df)):
        sgn = int(vals[j, col['signal']])
        if sgn == 0:
            continue
        vs, ss, ds = (int(vals[j, col['vol_state']]),
                      int(vals[j, col['sig_state']]),
                      int(vals[j, col['dir_state']]))
        if vs < 0 or ss < 0 or ds < 0:
            continue
        state = (vs, ss, ds)
        o, h, l, c = (vals[j, col['open']], vals[j, col['high']],
                      vals[j, col['low']], vals[j, col['close']])
        direction = 'R_up' if sgn == -1 else 'R_dn'
        mc = mc_cache.get((state, direction))
        ell = 0 if mc is None else mc['optimal_ell']
        is_mkt = True if mc is None else mc['is_market_order']
        if is_mkt or ell == 0:
            limit = fill = o; filled = True; missed = False; impr = 0.0; is_mkt = True
        else:
            if sgn == -1:
                limit = o + ell * tick_size; filled = h >= limit
            else:
                limit = o - ell * tick_size; filled = l <= limit
            if filled:
                fill = limit; missed = False
                impr = (fill - o) / tick_size if sgn == -1 else (o - fill) / tick_size
            else:
                fill = c; missed = True
                impr = (c - o) / tick_size if sgn == -1 else (o - c) / tick_size
        rec.append(dict(bar_time=idx[j], signal=sgn, state=state,
                        limit_level_ticks=ell, filled=filled,
                        improvement_ticks=impr, is_market_fallback=is_mkt,
                        is_missed_limit=missed))
    return pd.DataFrame(rec).set_index('bar_time') if rec else pd.DataFrame()


# ---------------- FIX #5: report both metrics, correctly labelled ----------------
def summarize(trades):
    lim = trades[~trades['is_market_fallback']]
    filled = lim[lim['filled']]
    missed = lim[lim['is_missed_limit']]
    fill_rate = lim['filled'].mean() if len(lim) else np.nan
    mean_impr_filled = filled['improvement_ticks'].mean() if len(filled) else np.nan
    return {
        'limit_orders': len(lim),
        'fill_rate': fill_rate,
        'mean_impr_filled': mean_impr_filled,
        'mean_cost_missed': missed['improvement_ticks'].mean() if len(missed) else np.nan,
        # Metric A: expected execution alpha vs a market order (README headline)
        'net_vs_market': (fill_rate * mean_impr_filled) if len(filled) else np.nan,
        # Metric B: realized avg across all limit trades incl. close fallback
        'realized_avg_all': lim['improvement_ticks'].mean() if len(lim) else np.nan,
    }
