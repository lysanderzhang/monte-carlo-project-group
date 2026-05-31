"""
make_test_data.py
=================

Generates SYNTHETIC futures data + AI-agent signal files in the EXACT on-disk
format the notebook's parsers expect, so you can run the whole pipeline
end-to-end without the real (large) CSVs.

On-disk format produced
------------------------
Contract CSV rows (>= 8 comma-separated fields; parser reads indices 0,1,2,3,7):
    excel_date , hour , minute , price , price , price , price , volume

AIAgent CSV rows (>= 5 fields; parser reads indices 0,1,2,3,4):
    excel_date , hour , minute , price , signed_position

This deliberately mirrors `_parse_contract` and `load_ai_agent_signal`:
  - one representative price per 1-minute row (open=high=low=close)
  - Excel serial date in column 0 (Excel epoch 1899-12-30)
  - AIAgent carries a *running signed position*, NOT an order

Run:
    python make_test_data.py            # writes into ./data/Test
"""

from __future__ import annotations
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path

EXCEL_EPOCH = datetime(1899, 12, 30)


def to_excel_serial(d: datetime) -> int:
    return int((d - EXCEL_EPOCH).days)


def _trading_minutes(day: date, n_minutes: int = 390):
    """US equity-index session 09:30 -> 16:00 == 390 one-minute slots."""
    start = datetime(day.year, day.month, day.day, 9, 30)
    return [start + timedelta(minutes=i) for i in range(n_minutes)]


def _business_days(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:          # Mon-Fri
            yield d
        d += timedelta(days=1)


def generate_contract(
    path: Path,
    start: date,
    end: date,
    base_price: float,
    tick_size: float,
    daily_vol_base: int,
    vol_regime_fn=None,
    drop_minute_prob: float = 0.0,
    seed: int = 0,
):
    """
    Write one synthetic contract CSV.

    vol_regime_fn(day_index) -> volatility multiplier, lets us inject a
    'COVID-like' spike so the volatility-state binning has something to chew on.

    drop_minute_prob simulates missing 1-minute bars (data gaps). This is the
    condition that breaks index-based (vs time-based) OHLC windowing, so we can
    test that the fix handles it.
    """
    rng = np.random.default_rng(seed)
    rows = []
    price = base_price
    days = list(_business_days(start, end))

    for di, day in enumerate(days):
        vmult = 1.0 if vol_regime_fn is None else float(vol_regime_fn(di))
        # per-minute log-return vol scales with regime multiplier
        minute_sigma = 0.0006 * vmult
        for minute in _trading_minutes(day):
            if drop_minute_prob and rng.random() < drop_minute_prob:
                continue  # simulate a missing minute
            ret = rng.normal(0.0, minute_sigma)
            price *= np.exp(ret)
            # snap to tick grid (futures trade on a tick lattice)
            snapped = round(price / tick_size) * tick_size
            vol = max(1, int(rng.poisson(daily_vol_base / 390 * (0.5 + vmult))))
            serial = to_excel_serial(minute)
            rows.append(
                f"{serial},{minute.hour},{minute.minute},"
                f"{snapped:.4f},{snapped:.4f},{snapped:.4f},{snapped:.4f},{vol}"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n")
    return len(rows)


def generate_aiagent(
    path: Path,
    start: date,
    end: date,
    base_price: float,
    active_prob: float = 0.07,
    max_units: int = 6,
    seed: int = 99,
):
    """
    Write a synthetic AIAgent CSV: a running signed position that occasionally
    scales in/out (so order = position.diff() is sparse, matching the ~7% non-flat
    rate noted in the README).
    """
    rng = np.random.default_rng(seed)
    rows = []
    position = 0
    price = base_price
    days = list(_business_days(start, end))
    for day in days:
        # AIAgent is recorded on 5-minute bars (matches TAU=5 default alignment)
        for minute in _trading_minutes(day, n_minutes=390)[::5]:
            price *= np.exp(rng.normal(0, 0.001))
            if rng.random() < active_prob:
                # change position by +/- 1..max_units, occasionally flip sign
                delta = rng.integers(1, max_units + 1) * rng.choice([-1, 1])
                position = int(np.clip(position + delta, -max_units, max_units))
            serial = to_excel_serial(minute)
            rows.append(f"{serial},{minute.hour},{minute.minute},{price:.2f},{position}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n")
    return len(rows)


def covid_like(di: int) -> float:
    """Volatility multiplier: calm -> spike around day ~45 -> elevated -> calmer."""
    if di < 35:
        return 1.0
    if di < 50:
        return 1.0 + (di - 35) * 0.9      # ramp up to the crash
    if di < 70:
        return 14.0 - (di - 50) * 0.45    # elevated, slowly decaying
    return 3.0


def build_test_suite(root: Path):
    """
    OPTIONAL offline smoke-test data. Each tier goes in its OWN folder so the
    notebook's auto-discovery treats it as a normal asset (one AIAgent file +
    contracts per folder). Use only when you don't have the real CSVs handy.

      SynthTiny  : 1 contract, ~10 business days, clean        -> fast smoke test
      SynthSmall : 2 contracts w/ rollover + 3% missing minutes -> stitch + gap fix
      SynthCovid : 1 contract, ~4 months, volatility spike      -> state binning
    """
    data = root / "data"

    # --- SynthTiny ---
    generate_contract(data / "SynthTiny" / "ZZH20.csv",
        date(2020, 1, 2), date(2020, 1, 15), 8800.0, 0.25, 200_000, seed=1)
    generate_aiagent(data / "SynthTiny" / "AIAgent_SynthTiny.csv",
        date(2020, 1, 2), date(2020, 1, 15), 8800.0, seed=11)

    # --- SynthSmall: two contracts that roll over, WITH missing minutes ---
    generate_contract(data / "SynthSmall" / "ZZH20.csv",
        date(2020, 1, 2), date(2020, 2, 20), 8800.0, 0.25, 200_000, drop_minute_prob=0.03, seed=2)
    generate_contract(data / "SynthSmall" / "ZZM20.csv",
        date(2020, 2, 3), date(2020, 3, 20), 8820.0, 0.25, 600_000, drop_minute_prob=0.03, seed=3)
    generate_aiagent(data / "SynthSmall" / "AIAgent_SynthSmall.csv",
        date(2020, 1, 2), date(2020, 3, 20), 8800.0, seed=22)

    # --- SynthCovid: single long contract with a volatility regime spike ---
    generate_contract(data / "SynthCovid" / "ZZH20.csv",
        date(2020, 1, 2), date(2020, 5, 29), 8800.0, 0.25, 300_000, vol_regime_fn=covid_like, seed=4)
    generate_aiagent(data / "SynthCovid" / "AIAgent_SynthCovid.csv",
        date(2020, 1, 2), date(2020, 5, 29), 8800.0, seed=44)

    print("Wrote synthetic assets: SynthTiny, SynthSmall, SynthCovid under", data)


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    build_test_suite(root)
