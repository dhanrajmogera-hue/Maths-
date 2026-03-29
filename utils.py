"""
=============================================================================
utils.py
Member 4 : [Altamash Ansari] — Theme Colors, Gauge, Input Parsing, Metric Cards
=============================================================================
Commits to make:
  Commit 1: "Added theme colors and matplotlib axis styling"
  Commit 2: "Added gauge chart, input parser and metric card HTML"
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import io

# ══════════════════════════════════════════════════════════════════════════════
#  DARK THEME COLORS
# ══════════════════════════════════════════════════════════════════════════════
C = {
    "bg"    : "#1a1a2e",  "panel" : "#16213e",
    "dark"  : "#e2e8f0",  "gray"  : "#94a3b8",
    "grid"  : "#2d3748",  "border": "#4a5568",
    "blue"  : "#60a5fa",  "red"   : "#f87171",
    "green" : "#4ade80",  "orange": "#fb923c",
    "purple": "#c084fc",  "teal"  : "#2dd4bf",
    "yellow": "#fbbf24",
}

# ══════════════════════════════════════════════════════════════════════════════
#  AXIS STYLING
# ══════════════════════════════════════════════════════════════════════════════
def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(C["panel"])
    ax.set_title(title, fontsize=10, fontweight="bold", color=C["dark"], pad=7)
    ax.set_xlabel(xlabel, fontsize=8, color=C["gray"])
    ax.set_ylabel(ylabel, fontsize=8, color=C["gray"])
    ax.tick_params(colors=C["gray"], labelsize=7)
    ax.grid(True, color=C["grid"], lw=0.8, ls="--", alpha=0.8)
    for sp in ax.spines.values():
        sp.set_edgecolor(C["border"])

def smart_xticks(ax, idx, labels, max_show=12):
    n    = len(idx)
    step = max(1, n // max_show)
    ax.set_xticks(idx[::step])
    ax.set_xticklabels(labels[::step],
                       fontsize=max(5, 8 - n // 6),
                       rotation=30, ha="right")

def bar_labels(ax, idx, vals, fmt="{:.1f}"):
    if len(idx) > 15:
        return
    mx = max(abs(vals)) if max(abs(vals)) > 0 else 1
    for i, val in enumerate(vals):
        off = mx * 0.04 if val >= 0 else -mx * 0.12
        ax.text(idx[i], val + off, fmt.format(val),
                ha="center", fontsize=max(5, 7 - len(idx) // 5),
                fontweight="bold", color=C["dark"])

# ══════════════════════════════════════════════════════════════════════════════
#  GAUGE  (0 to 1)  — used for both |R| and R²
# ══════════════════════════════════════════════════════════════════════════════
def draw_gauge(ax, value, title, top_label, cmap="RdYlGn"):
    ax.set_facecolor(C["panel"])
    ax.set_title(title, fontsize=10, fontweight="bold", color=C["dark"], pad=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(C["border"])
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.62, 1.62)
    ax.axis("off")
    grad = np.linspace(0, 1, 500).reshape(1, -1)
    ax.imshow(grad, extent=[0, 1, -0.08, 0.22],
              aspect="auto", cmap=cmap, vmin=0, vmax=1, zorder=1)
    ax.add_patch(plt.Rectangle((0, -0.08), 1, 0.30,
                 fill=False, edgecolor=C["border"], lw=1.2, zorder=2))
    ax.axvline(value, ymin=0.38, ymax=0.68, color="white", lw=3.5, zorder=5)
    ax.annotate("", xy=(value, 0.30), xytext=(value, 0.54),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2.2))
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        ax.text(t, -0.30, str(t), ha="center", fontsize=8, color=C["gray"])
    ax.text(0.5, -0.48, "Weak ←──────────────→ Strong",
            ha="center", fontsize=6.5, color=C["gray"])
    ax.text(value, 0.84, f"{value:.4f}",
            ha="center", fontsize=14, fontweight="bold", color="white")
    ax.text(value, 1.28, top_label,
            ha="center", fontsize=8.5, fontweight="bold", color=C["green"])

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE → PNG BUFFER
# ══════════════════════════════════════════════════════════════════════════════
def fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140,
                bbox_inches="tight", facecolor=C["bg"])
    buf.seek(0)
    return buf

# ══════════════════════════════════════════════════════════════════════════════
#  INPUT PARSING
# ══════════════════════════════════════════════════════════════════════════════
def parse_values(text):
    """Convert comma-separated text to list of floats."""
    try:
        vals = [float(v.strip())
                for v in text.replace("\n", ",").split(",") if v.strip()]
        return vals, None
    except Exception:
        return None, "⚠ Invalid — use comma-separated numbers e.g.  10, 20, 30"

# ══════════════════════════════════════════════════════════════════════════════
#  METRIC CARD HTML
# ══════════════════════════════════════════════════════════════════════════════
def metric_card(label, value, sub=""):
    s = (f'<div class="metric-card">'
         f'<div class="lbl">{label}</div>'
         f'<div class="val">{value}</div>')
    if sub:
        s += f'<div class="sub">{sub}</div>'
    return s + '</div>'

# ══════════════════════════════════════════════════════════════════════════════
#  INTERPRETATION LABELS
# ══════════════════════════════════════════════════════════════════════════════
def label_r(r):
    """Interpret absolute Spearman R value (0 to 1)."""
    if r >= 0.99:  return "Perfect"
    elif r >= 0.9: return "Very Strong"
    elif r >= 0.7: return "Strong"
    elif r >= 0.5: return "Moderate"
    elif r >= 0.3: return "Weak"
    else:          return "Very Weak / No"

def label_r2(r2):
    """Interpret R² goodness of fit (0 to 1)."""
    if r2 >= 0.99:   return "Excellent"
    elif r2 >= 0.95: return "Very Good"
    elif r2 >= 0.85: return "Good"
    elif r2 >= 0.70: return "Moderate"
    else:            return "Poor"
