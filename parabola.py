"""
=============================================================================
parabola.py
Member 3 : [Your Name] — Parabolic Curve Fitting Logic + 6 Panel Graph
=============================================================================
Commits to make:
  Commit 1: "Added compute_parabola() with normal equations and matrix solve"
  Commit 2: "Added 6-panel parabola graph with residuals and R² gauge"
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils import C, style_ax, smart_xticks, bar_labels, draw_gauge, label_r2


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — BUILD WORKING TABLE COLUMNS
# ══════════════════════════════════════════════════════════════════════════════

def build_table(x, y):
    """
    Compute all columns needed for the working table and normal equations.

    Columns: x, y, x², x³, x⁴, xy, x²y
    Sums:    Σx, Σy, Σx², Σx³, Σx⁴, Σxy, Σx²y
    """
    x2  = x ** 2
    x3  = x ** 3
    x4  = x ** 4
    xy  = x * y
    x2y = x2 * y
    return dict(
        x2=x2, x3=x3, x4=x4, xy=xy, x2y=x2y,
        sx=x.sum(),   sx2=x2.sum(), sx3=x3.sum(), sx4=x4.sum(),
        sy=y.sum(),   sxy=xy.sum(), sx2y=x2y.sum()
    )


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — SOLVE NORMAL EQUATIONS USING MATRIX METHOD
# ══════════════════════════════════════════════════════════════════════════════

def compute_parabola(x, y):
    """
    Fit parabola  y = a + bx + cx²  using Least Squares Method.

    Normal Equations (derived by minimizing Σ(y − ŷ)²):
        Σy   = na    + b·Σx   + c·Σx²
        Σxy  = a·Σx  + b·Σx²  + c·Σx³
        Σx²y = a·Σx² + b·Σx³  + c·Σx⁴

    Matrix Form:  A · [a, b, c]ᵀ = B
    Solved using: numpy.linalg.solve(A, B)

    Also computes:
        ŷ        = fitted values
        residuals = y − ŷ
        R²       = 1 − SS_res/SS_tot   (goodness of fit, 0 to 1)
    """
    n   = len(x)
    t   = build_table(x, y)
    sx, sx2, sx3, sx4 = t["sx"], t["sx2"], t["sx3"], t["sx4"]
    sy, sxy, sx2y     = t["sy"], t["sxy"], t["sx2y"]

    # Set up matrix A and vector B
    A = np.array([
        [n,   sx,  sx2],
        [sx,  sx2, sx3],
        [sx2, sx3, sx4]
    ])
    B = np.array([sy, sxy, sx2y])

    # Solve for a, b, c
    a, b, c = np.linalg.solve(A, B)

    # Fitted values and residuals
    y_fit = a + b * x + c * x**2
    res_v = y - y_fit

    # Goodness of fit
    ss_res = (res_v**2).sum()
    ss_tot = ((y - y.mean())**2).sum()
    R2     = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0

    return dict(
        a=a, b=b, c=c,
        y_fit=y_fit, res_v=res_v,
        ss_res=ss_res, ss_tot=ss_tot, R2=R2,
        table=t, n=n
    )


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — 6-PANEL GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def plot_parabola(x, y, res, lx, ly):
    """
    Generate 6-panel Parabola analysis graph:
      Panel 1 — Data points + smooth fitted parabola
      Panel 2 — Residual bars per point
      Panel 3 — Actual vs Fitted scatter
      Panel 4 — Residual histogram
      Panel 5 — Actual vs Fitted line comparison
      Panel 6 — R² gauge (0 to 1)
    """
    a, b, c   = res["a"], res["b"], res["c"]
    y_fit     = res["y_fit"]
    res_v     = res["res_v"]
    R2        = res["R2"]
    ss_res    = res["ss_res"]
    n   = len(x)
    idx = np.arange(1, n + 1)
    lbl = np.array([f"P{i}" for i in idx])
    bw  = max(0.3, min(0.75, 6.0 / n))
    ms  = max(3, 7 - n // 6)
    sb  = "+" if b >= 0 else "−"
    sc  = "+" if c >= 0 else "−"
    eq  = f"y={a:.3f} {sb} {abs(b):.3f}x {sc} {abs(c):.4f}x²"

    # Smooth curve for plotting
    xs = np.linspace(x.min() - abs(x.max() - x.min()) * 0.1,
                     x.max() + abs(x.max() - x.min()) * 0.1, 500)
    ys = a + b * xs + c * xs**2

    fig = plt.figure(figsize=(17, 9))
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle(
        f"Parabolic Curve Fitting  (y = a + bx + cx²)  |  {lx} vs {ly}  "
        f"(n={n})  |  R² = {R2:.4f}  →  {label_r2(R2)}",
        fontsize=12, fontweight="bold", color=C["dark"], y=0.99)

    gs   = gridspec.GridSpec(2, 3, hspace=0.55, wspace=0.42)
    axes = [fig.add_subplot(gs[r, c_]) for r in range(2) for c_ in range(3)]

    # ── Panel 1: Data + parabola ──────────────────────────────────────────────
    style_ax(axes[0], "Data & Fitted Parabola", lx, ly)
    axes[0].scatter(x, y, color=C["blue"], s=70, zorder=5,
                    edgecolors=C["bg"], lw=0.8, label="Observed")
    axes[0].plot(xs, ys, color=C["red"], lw=2.2, label=eq)
    axes[0].fill_between(xs, ys,
                         min(ys.min(), y.min()) - abs(y.max() - y.min()) * 0.05,
                         alpha=0.08, color=C["red"])
    if n <= 20:
        for i in range(n):
            axes[0].annotate(f"P{i+1}", (x[i], y[i]),
                             textcoords="offset points", xytext=(5, 3),
                             fontsize=6, color=C["gray"])
    axes[0].legend(fontsize=7, facecolor=C["panel"], labelcolor=C["dark"])

    # ── Panel 2: Residual bars ────────────────────────────────────────────────
    style_ax(axes[1], "Residuals per Point", "Point", f"{ly} − ŷ")
    axes[1].bar(idx, res_v,
                color=[C["red"] if v < 0 else C["green"] for v in res_v],
                edgecolor=C["bg"], width=bw, lw=0.4)
    axes[1].axhline(0, color=C["dark"], lw=1.2)
    smart_xticks(axes[1], idx, lbl)
    ypad = max(abs(res_v)) * 0.22 if max(abs(res_v)) > 0 else 0.5
    axes[1].set_ylim(res_v.min() - ypad, res_v.max() + ypad)
    bar_labels(axes[1], idx, res_v, fmt="{:+.2f}")

    # ── Panel 3: Actual vs Fitted scatter ─────────────────────────────────────
    style_ax(axes[2], "Actual vs Fitted", f"Actual {ly}", "Fitted ŷ")
    axes[2].scatter(y, y_fit, color=C["purple"], s=65, zorder=5,
                    edgecolors=C["bg"], lw=0.8)
    lo_ = min(y.min(), y_fit.min()) - abs(y.max() - y.min()) * 0.05
    hi_ = max(y.max(), y_fit.max()) + abs(y.max() - y.min()) * 0.05
    axes[2].plot([lo_, hi_], [lo_, hi_], color=C["orange"],
                 lw=1.6, ls="--", label="Perfect fit")
    axes[2].set_xlim(lo_, hi_)
    axes[2].set_ylim(lo_, hi_)
    axes[2].legend(fontsize=7, facecolor=C["panel"], labelcolor=C["dark"])

    # ── Panel 4: Residual histogram ───────────────────────────────────────────
    style_ax(axes[3], "Residual Distribution", "Residual", "Frequency")
    bins = max(4, min(n, int(1 + 3.322 * np.log10(n))))
    axes[3].hist(res_v, bins=bins, color=C["blue"],
                 edgecolor=C["bg"], lw=0.6, alpha=0.85)
    axes[3].axvline(0, color=C["red"], lw=1.6, ls="--", label="Zero")
    axes[3].axvline(res_v.mean(), color=C["yellow"], lw=1.4, ls="-.",
                    label=f"Mean = {res_v.mean():.3f}")
    axes[3].legend(fontsize=7, facecolor=C["panel"], labelcolor=C["dark"])

    # ── Panel 5: Actual vs Fitted line ────────────────────────────────────────
    style_ax(axes[4], f"Actual vs Fitted  ({ly})", "Point", ly)
    axes[4].plot(idx, y,     "o-",  color=C["blue"], lw=1.8, ms=ms,
                 markeredgecolor=C["bg"], label="Actual")
    axes[4].plot(idx, y_fit, "s--", color=C["red"],  lw=1.8, ms=ms,
                 markeredgecolor=C["bg"], label="Fitted ŷ")
    axes[4].fill_between(idx, y, y_fit, alpha=0.15, color=C["teal"])
    smart_xticks(axes[4], idx, lbl)
    axes[4].legend(fontsize=7, facecolor=C["panel"], labelcolor=C["dark"])

    # ── Panel 6: R² gauge ─────────────────────────────────────────────────────
    draw_gauge(axes[5], R2,
               "R² Goodness of Fit  (0 to 1)",
               f"{label_r2(R2)} Fit")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig
