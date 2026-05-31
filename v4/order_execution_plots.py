"""
oe_plots.py
===========

Plotting helpers for order_execution.  Each takes a `run_pipeline` result dict
plus the Config, so the notebook can render every figure in one line per asset.
"""

from itertools import product

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PALETTE = sns.color_palette("tab10")


def setup_style():
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.3, "font.size": 11})


def plot_execution_frontiers(res, cfg, ref_volume=1, ref_dir=1):
    """Fill-probability frontiers across the three volatility states."""
    cache, asset = res["mc_cache"], res["asset"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = {0: "Low Vol", 1: "Med Vol", 2: "High Vol"}
    for ss, color in zip(range(cfg.n_states), PALETTE[:3]):
        state = (ref_volume, ss, ref_dir)
        for k, direction in enumerate(["R_up", "R_dn"]):
            r, ax = cache[(state, direction)], axes[k]
            x, y = r["ell_grid"], r["fill_rate"]
            valid = y >= 0.005
            last = np.where(valid)[0].max() + 1 if valid.any() else len(x)
            ax.plot(x[:last], y[:last], color=color, label=labels[ss])
            ax.fill_between(x[:last], r["ci_fill_lo"][:last], r["ci_fill_hi"][:last],
                            color=color, alpha=0.15)
            ax.axhline(cfg.fill_prob_threshold, color="k", ls="--", lw=1)
    axes[0].set(title="Execution Frontier — Sell limit (R_up)",
                xlabel="Limit level (ticks above open)", ylabel="Fill probability")
    axes[1].set(title="Execution Frontier — Buy limit (R_dn)",
                xlabel="Limit level (ticks below open)", ylabel="Fill probability")
    for ax in axes:
        ax.legend()
        ax.set_ylim(0, 1.05)
    fig.suptitle(f"{asset} — MC Execution Frontiers "
                 f"(med volume, varying volatility, flat trend, tau={cfg.tau}min)", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_figure2(res, cfg, states_to_show=None, direction="R_dn"):
    """Figure-2 replication: counts, fill counts, ePDF, survival."""
    store, asset = res["store"], res["asset"]
    states_to_show = states_to_show or [(1, 0, 1), (1, 1, 1), (1, 2, 1)]
    labels = ["Low Vol", "Med Vol", "High Vol"][:len(states_to_show)]
    colors = PALETTE[:len(states_to_show)]
    max_ell = min(40, max(len(store.get_pmf(s, direction)) for s in states_to_show))

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)
    ax_tl, ax_bl = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    ax_tr, ax_br = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    x, bw = np.arange(max_ell), 0.8 / len(states_to_show)

    def _pad(a):
        return np.pad(a[:max_ell], (0, max(0, max_ell - len(a))))[:max_ell]

    for i, (st, lab, col) in enumerate(zip(states_to_show, labels, colors)):
        counts = store._counts[st][direction]
        c = np.zeros(max_ell)
        c[:min(len(counts), max_ell)] = counts[:max_ell]
        pmf = _pad(store.get_pmf(st, direction))
        surv = _pad(store.get_survival(st, direction))
        fillc = np.array([c[e:].sum() for e in range(max_ell)])
        off = (i - len(states_to_show) / 2 + 0.5) * bw
        ax_tl.bar(x + off, c, width=bw, color=col, alpha=0.7, label=lab)
        ax_bl.bar(x + off, pmf, width=bw, color=col, alpha=0.7)
        ax_tr.bar(x + off, fillc, width=bw, color=col, alpha=0.7)
        ax_br.bar(x + off, surv, width=bw, color=col, alpha=0.7)

    dl = "R_up (Sell)" if direction == "R_up" else "R_dn (Buy)"
    ax_tl.set(title="Counts/Frequencies", xlabel="Spreads", ylabel="Count")
    ax_tl.legend(fontsize=9)
    ax_bl.set(title=f"ePDF of {dl}", xlabel="Spreads", ylabel="Probability")
    ax_tr.set(title="Counts of getting filled", xlabel="Spreads", ylabel="Count")
    ax_br.set(title="Fill probability P(X >= l)", xlabel="Spreads", ylabel="Probability")
    fig.suptitle(f"{asset} — Figure 2 Replication (tau={cfg.tau}min)", fontsize=13)
    plt.show()


def plot_cumulative_improvement(res, cfg):
    """Cumulative gross vs net improvement and the rolling fill rate."""
    trades, asset = res["trades"], res["asset"]
    lim = trades[~trades["is_market_fallback"]].copy()
    filled = lim[lim["filled"]]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    axes[0].plot(filled.index, filled["improvement_ticks"].cumsum(),
                 color=PALETTE[0], label="Gross (filled only)")
    axes[0].plot(lim.index, lim["improvement_ticks"].cumsum(),
                 color=PALETTE[1], ls="--", label="Net (incl. missed fallbacks)")
    axes[0].legend(fontsize=9)
    axes[0].set(title=f"{asset} — Cumulative Improvement: Gross vs Net",
                ylabel="Cumulative ticks vs open")
    axes[1].plot(lim.index, lim["filled"].rolling(25).mean(), color=PALETTE[1])
    axes[1].axhline(cfg.fill_prob_threshold, color="k", ls="--", lw=1,
                    label=f"theta={cfg.fill_prob_threshold}")
    axes[1].set(title="Rolling Fill Rate (25-bar)", ylabel="Fill rate", xlabel="Date")
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def plot_calibration(res, cfg):
    """Predicted (ePDF survival) vs realised fill rate per (state, level)."""
    trades, store, asset = res["trades"], res["store"], res["asset"]
    lim = trades[~trades["is_market_fallback"]]
    rec = []
    for (st, d), g in lim.groupby(["state", "direction"]):
        for ell, sub in g.groupby("limit_level_ticks"):
            if len(sub) < 5:
                continue
            rec.append({"predicted": store.fill_prob(st, d, int(ell)),
                        "realized": sub["filled"].mean(), "n": len(sub)})
    calib = pd.DataFrame(rec)
    if calib.empty:
        print("Not enough (state, level) groups for a calibration plot.")
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(calib["predicted"], calib["realized"], s=calib["n"] * 2,
               alpha=0.6, color=PALETTE[2])
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set(title=f"{asset} — Fill Probability Calibration",
           xlabel="Predicted (ePDF survival)", ylabel="Realized fill rate",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_sensitivity(sens_df, asset):
    """Three heatmaps: fill rate, gross edge, realised net over (tau, threshold)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    sns.heatmap(sens_df.pivot(index="tau", columns="threshold", values="fill_rate"),
                annot=True, fmt=".2f", cmap="YlGn", ax=axes[0],
                cbar_kws={"label": "Fill rate"})
    sns.heatmap(sens_df.pivot(index="tau", columns="threshold", values="gross_edge"),
                annot=True, fmt=".2f", cmap="YlOrRd", ax=axes[1],
                cbar_kws={"label": "Gross edge (fills only)"})
    sns.heatmap(sens_df.pivot(index="tau", columns="threshold", values="net_vs_market"),
                annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=axes[2],
                cbar_kws={"label": "Net incl. misses"})
    axes[0].set(title=f"{asset} — Fill Rate")
    axes[1].set(title=f"{asset} — Gross Edge")
    axes[2].set(title=f"{asset} — Realized Net")
    fig.suptitle("Sensitivity to tau and fill threshold", y=1.03)
    plt.tight_layout()
    plt.show()
