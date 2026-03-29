"""
=============================================================================
spearman.py
Member 3 : [Vishal Prajapati] — Spearman Rank Correlation Logic + 6 Panel Graph
=============================================================================
Commits to make:
  Commit 1: "Added assign_ranks() with tie handling and correction factor"
  Commit 2: "Added compute_spearman() and 6-panel Spearman graph"
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils import C, style_ax, smart_xticks, bar_labels, draw_gauge, label_r


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — ASSIGN RANKS (with tie handling)
# ══════════════════════════════════════════════════════════════════════════════

def assign_ranks(data):
    """
    Assign ranks to data array.
    Tied (repeated) values get the AVERAGE of the ranks they would occupy.

    Example 1:  [10, 20, 20, 30]  →  ranks [1.0, 2.5, 2.5, 4.0]
                 20 appears at positions 2 & 3 → avg rank = (2+3)/2 = 2.5

    Example 2:  [5, 5, 5, 10]   →  ranks [2.0, 2.0, 2.0, 4.0]
                 5 appears at positions 1,2,3 → avg rank = (1+2+3)/3 = 2.0
    """
    data  = np.array(data, dtype=float)
    n     = len(data)
    ranks = np.empty(n)
    sidx  = np.argsort(data)          # indices that sort data ascending
    i = 0
    while i < n:
        j = i
        # Walk forward while values are equal (find entire tied group)
        while j < n - 1 and data[sidx[j]] == data[sidx[j + 1]]:
            j += 1
        # Average rank for the tied group
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[sidx[k]] = avg_rank
        i = j + 1
    return ranks


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — DETECT TIES
# ══════════════════════════════════════════════════════════════════════════════

def detect_ties(data):
    """Return dict of {value: count} for all repeated values."""
    vals, counts = np.unique(data, return_counts=True)
    return {float(v): int(c) for v, c in zip(vals, counts) if c > 1}


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — CORRECTION FACTOR T
# ══════════════════════════════════════════════════════════════════════════════

def correction_factor(data):
    """
    Tie correction factor:  T = Σ (t³ − t) / 12
    where t = number of observations in each tied group.

    Applied in corrected Spearman formula when repeated values exist.
    """
    _, counts = np.unique(data, return_counts=True)
    return sum((t**3 - t) / 12 for t in counts if t > 1)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — COMPUTE SPEARMAN R (with corrected formula)
# ══════════════════════════════════════════════════════════════════════════════

def compute_spearman(x, y):
    """
    Full Spearman Rank Correlation computation.

    Formulas used:
    ─────────────
    Standard (no ties):
        R = 1 − (6 Σd²) / (n(n²−1))

    Corrected (with ties):
        Ax = (n³−n)/12 − Tx
        Ay = (n³−n)/12 − Ty
        R  = (Ax + Ay − Σd²) / (2 √(Ax · Ay))

    where Tx = Σ(t³−t)/12 for tied groups in X
          Ty = Σ(t³−t)/12 for tied groups in Y

    Returns |R| (0 to 1) for strength and direction separately.
    """
    n  = len(x)
    Rx = assign_ranks(x)
    Ry = assign_ranks(y)
    d  = Rx - Ry
    d2 = d ** 2

    ties_x   = detect_ties(x)
    ties_y   = detect_ties(y)
    has_ties = bool(ties_x or ties_y)

    # Standard formula (ignores ties)
    R_std = 1 - (6 * d2.sum()) / (n * (n**2 - 1))

    # Corrected formula (accurate when ties exist)
    Tx    = correction_factor(x)
    Ty    = correction_factor(y)
    Ax    = (n**3 - n) / 12 - Tx
    Ay    = (n**3 - n) / 12 - Ty
    denom = 2 * np.sqrt(Ax * Ay)
    R_corr = (Ax + Ay - d2.sum()) / denom if denom != 0 else 0.0

    # Pearson on ranks — ground truth, always matches corrected formula
    mx, my = Rx.mean(), Ry.mean()
    num    = np.sum((Rx - mx) * (Ry - my))
    den    = np.sqrt(np.sum((Rx - mx)**2) * np.sum((Ry - my)**2))
    R_raw  = num / den if den != 0 else 0.0

    R_abs     = abs(R_raw)                                   # 0 to 1
    direction = "Positive ↑" if R_raw >= 0 else "Negative ↓"

    return dict(
        Rx=Rx, Ry=Ry, d=d, d2=d2,
        R_std=R_std, R_corr=R_corr, R_raw=R_raw, R_abs=R_abs,
        direction=direction,
        ties_x=ties_x, ties_y=ties_y,
        Tx=Tx, Ty=Ty, Ax=Ax, Ay=Ay,
        has_ties=has_ties
    )


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — 6-PANEL GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def plot_spearman(x, y, res, lx, ly):
    """
    Generate 6-panel Spearman analysis graph:
      Panel 1 — Raw scatter + trend line
      Panel 2 — Rank scatter (Rank X vs Rank Y)
      Panel 3 — d bar chart (rank differences)
      Panel 4 — d² bar chart (squared differences)
      Panel 5 — Rank distribution line comparison
      Panel 6 — |R| gauge (0 to 1)
    """
    Rx, Ry = res["Rx"], res["Ry"]
    d,  d2  = res["d"],  res["d2"]
    R_abs   = res["R_abs"]
    n   = len(x)
    idx = np.arange(1, n + 1)
    lbl = np.array([f"P{i}" for i in idx])
    bw  = max(0.3, min(0.75, 6.0 / n))     # dynamic bar width
    ms  = max(3, 7 - n // 6)               # dynamic marker size

    fig = plt.figure(figsize=(17, 9))
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle(
        f"Spearman Rank Correlation  |  {lx} vs {ly}  "
        f"(n={n})  |  |R| = {R_abs:.4f}  →  {label_r(R_abs)}  ({res['direction']})",
        fontsize=12, fontweight="bold", color=C["dark"], y=0.99)

    gs   = gridspec.GridSpec(2, 3, hspace=0.55, wspace=0.42)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    # ── Panel 1: Raw scatter ──────────────────────────────────────────────────
    style_ax(axes[0], f"Raw Data: {lx} vs {ly}", lx, ly)
    axes[0].scatter(x, y, color=C["blue"], s=65, zorder=5,
                    edgecolors=C["bg"], lw=0.8, label="Data")
    if n <= 20:
        for i in range(n):
            axes[0].annotate(f"P{i+1}", (x[i], y[i]),
                             textcoords="offset points", xytext=(5, 3),
                             fontsize=6, color=C["gray"])
    m_, b_ = np.polyfit(x, y, 1)
    xl = np.linspace(x.min() - 0.5, x.max() + 0.5, 300)
    axes[0].plot(xl, m_ * xl + b_, color=C["red"],
                 lw=1.8, ls="--", label="Trend")
    axes[0].legend(fontsize=7, facecolor=C["panel"], labelcolor=C["dark"])

    # ── Panel 2: Rank scatter ─────────────────────────────────────────────────
    style_ax(axes[1], f"Rank Scatter: Rank({lx}) vs Rank({ly})",
             f"Rank({lx})", f"Rank({ly})")
    axes[1].scatter(Rx, Ry, color=C["purple"], s=65, zorder=5,
                    edgecolors=C["bg"], lw=0.8)
    if n <= 20:
        for i in range(n):
            axes[1].annotate(f"P{i+1}", (Rx[i], Ry[i]),
                             textcoords="offset points", xytext=(5, 3),
                             fontsize=6, color=C["gray"])
    diag = np.linspace(1, n, 200)
    axes[1].plot(diag, diag, color=C["green"], lw=1.4,
                 ls="--", alpha=0.7, label="Perfect |R|=1")
    axes[1].legend(fontsize=7, facecolor=C["panel"], labelcolor=C["dark"])

    # ── Panel 3: d bar ────────────────────────────────────────────────────────
    style_ax(axes[2], "Rank Difference  d = Rx − Ry", "Point", "d")
    axes[2].bar(idx, d,
                color=[C["red"] if v < 0 else C["blue"] for v in d],
                edgecolor=C["bg"], width=bw, lw=0.4)
    axes[2].axhline(0, color=C["dark"], lw=1.2)
    smart_xticks(axes[2], idx, lbl)
    bar_labels(axes[2], idx, d, fmt="{:+.1f}")

    # ── Panel 4: d² bar ───────────────────────────────────────────────────────
    style_ax(axes[3], "Squared Differences  d²", "Point", "d²")
    axes[3].bar(idx, d2, color=C["orange"], edgecolor=C["bg"], width=bw, lw=0.4)
    if d2.mean() > 0:
        axes[3].axhline(d2.mean(), color=C["red"], lw=1.4, ls="--",
                        label=f"Mean d² = {d2.mean():.2f}")
        axes[3].legend(fontsize=7, facecolor=C["panel"], labelcolor=C["dark"])
    smart_xticks(axes[3], idx, lbl)
    bar_labels(axes[3], idx, d2, fmt="{:.1f}")

    # ── Panel 5: Rank distribution ────────────────────────────────────────────
    style_ax(axes[4], "Rank Distribution Comparison", "Point", "Rank")
    axes[4].plot(idx, Rx, "o-", color=C["blue"],   lw=1.8, ms=ms,
                 markeredgecolor=C["bg"], label=f"Rank {lx}")
    axes[4].plot(idx, Ry, "s-", color=C["purple"], lw=1.8, ms=ms,
                 markeredgecolor=C["bg"], label=f"Rank {ly}")
    axes[4].fill_between(idx, Rx, Ry, alpha=0.15, color=C["teal"])
    smart_xticks(axes[4], idx, lbl)
    axes[4].legend(fontsize=7, facecolor=C["panel"], labelcolor=C["dark"])

    # ── Panel 6: |R| gauge ────────────────────────────────────────────────────
    draw_gauge(axes[5], R_abs,
               "|R| Strength Gauge  (0 to 1)",
               f"{label_r(R_abs)}  ({res['direction']})")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig
