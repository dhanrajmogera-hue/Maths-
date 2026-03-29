"""
=============================================================================
Assignment 9 — EM-4 (BSC07) | INFT Engineering
Topic  : Spearman Rank Correlation & Parabolic Curve Fitting
Run    : streamlit run app.py
Input  : CSV Upload  OR  Manual Entry

Description :
This project implements statistical methods including Spearman Rank
Correlation and Parabolic Curve Fitting with visualization.

Features :
- Handles CSV file input and manual data entry
- Automatically detects tied values and applies correction
- Displays step-by-step calculation and graphical analysis
- Interactive UI using Streamlit
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st
import pandas as pd
import io

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="EM-4 | Statistical Methods",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

.main-header {
    background: linear-gradient(90deg,#667eea 0%,#764ba2 100%);
    border-radius:16px; padding:26px 34px; margin-bottom:22px;
    box-shadow:0 8px 32px rgba(102,126,234,0.4);
}
.main-header h1 { font-family:'Space Mono',monospace; color:white; font-size:1.8rem; margin:0 0 6px 0; }
.main-header p  { color:rgba(255,255,255,0.72); font-size:0.90rem; margin:0; }

.section-title {
    font-family:'Space Mono',monospace; color:rgba(255,255,255,0.9);
    font-size:0.95rem; font-weight:700;
    border-left:4px solid #667eea; padding-left:12px; margin:20px 0 12px 0;
}

.result-badge {
    display:inline-block;
    background:linear-gradient(90deg,#11998e,#38ef7d);
    color:#0f2027; font-family:'Space Mono',monospace; font-weight:700;
    font-size:0.95rem; padding:10px 22px; border-radius:50px; margin:6px 4px;
    box-shadow:0 4px 18px rgba(56,239,125,0.35);
}
.result-badge-blue   { background:linear-gradient(90deg,#2193b0,#6dd5ed); color:#0f2027; }
.result-badge-orange { background:linear-gradient(90deg,#f7971e,#ffd200); color:#0f2027; }

.metric-row { display:flex; gap:10px; flex-wrap:wrap; margin:14px 0; }
.metric-card {
    background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.11);
    border-radius:12px; padding:13px 18px; min-width:130px; flex:1;
}
.metric-card .lbl { color:rgba(255,255,255,0.48); font-size:0.68rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px; }
.metric-card .val { color:white; font-family:'Space Mono',monospace; font-size:1.1rem; font-weight:700; }
.metric-card .sub { color:#38ef7d; font-size:0.74rem; margin-top:2px; }

.tie-box {
    background:rgba(251,191,36,0.10); border:1px solid rgba(251,191,36,0.35);
    border-radius:10px; padding:13px 17px; margin:10px 0;
    color:rgba(255,255,255,0.88); font-size:0.85rem; line-height:1.65;
}
.info-box {
    background:rgba(102,126,234,0.13); border:1px solid rgba(102,126,234,0.28);
    border-radius:10px; padding:13px 17px; margin:10px 0;
    color:rgba(255,255,255,0.82); font-size:0.85rem; line-height:1.65;
}
.formula-box {
    background:rgba(56,239,125,0.08); border:1px solid rgba(56,239,125,0.30);
    border-radius:10px; padding:14px 18px; margin:10px 0;
    color:rgba(255,255,255,0.92); font-size:0.88rem; line-height:1.75;
}
.upload-box {
    background:rgba(255,255,255,0.04); border:2px dashed rgba(102,126,234,0.5);
    border-radius:12px; padding:18px; margin:10px 0;
}

.stTextInput>label,.stNumberInput>label,.stSelectbox>label {
    color:rgba(255,255,255,0.8) !important; font-weight:500;
}
.stTextInput input,.stNumberInput input {
    background:rgba(255,255,255,0.07) !important;
    border:1px solid rgba(255,255,255,0.14) !important;
    color:white !important; border-radius:8px !important;
}
.stTextArea textarea {
    background:rgba(255,255,255,0.07) !important;
    border:1px solid rgba(255,255,255,0.14) !important;
    color:white !important; border-radius:8px !important;
    font-family:'Space Mono',monospace !important; font-size:0.85rem !important;
}
.stButton>button {
    background:linear-gradient(90deg,#667eea,#764ba2) !important;
    color:white !important; border:none !important; border-radius:10px !important;
    font-weight:600 !important; font-size:1rem !important;
    padding:10px 28px !important; width:100%;
    box-shadow:0 4px 14px rgba(102,126,234,0.4) !important;
}
.stTabs [data-baseweb="tab"]   { color:rgba(255,255,255,0.5) !important; }
.stTabs [aria-selected="true"] { color:white !important; font-weight:700; }
[data-testid="stFileUploader"] label { color:rgba(255,255,255,0.8) !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MATPLOTLIB DARK THEME
# ══════════════════════════════════════════════════════════════════════════════
C = {
    "bg":"#1a1a2e","panel":"#16213e","dark":"#e2e8f0","gray":"#94a3b8",
    "grid":"#2d3748","border":"#4a5568","blue":"#60a5fa","red":"#f87171",
    "green":"#4ade80","orange":"#fb923c","purple":"#c084fc",
    "teal":"#2dd4bf","yellow":"#fbbf24",
}

def style_ax(ax, title, xlabel, ylabel):
    """
    Applies consistent dark theme styling to matplotlib axes.
    """
    ax.set_facecolor(C["panel"])
    ax.set_title(title, fontsize=10, fontweight="bold", color=C["dark"], pad=7)
    ax.set_xlabel(xlabel, fontsize=8, color=C["gray"])
    ax.set_ylabel(ylabel, fontsize=8, color=C["gray"])
    ax.tick_params(colors=C["gray"], labelsize=7)
    ax.grid(True, color=C["grid"], lw=0.8, ls="--", alpha=0.8)
    for sp in ax.spines.values(): sp.set_edgecolor(C["border"])

def smart_xticks(ax, idx, labels, max_show=12):
    """
    Dynamically adjusts x-axis ticks to avoid clutter in large datasets.
    """
    n    = len(idx)
    step = max(1, n // max_show)
    ax.set_xticks(idx[::step])
    ax.set_xticklabels(labels[::step], fontsize=max(5,8-n//6),
                       rotation=30, ha="right")

def bar_labels(ax, idx, vals, fmt="{:.1f}"):
    if len(idx) > 15: return
    mx = max(abs(vals)) if max(abs(vals)) > 0 else 1
    for i, val in enumerate(vals):
        off = mx*0.04 if val >= 0 else -mx*0.12
        ax.text(idx[i], val+off, fmt.format(val),
                ha="center", fontsize=max(5, 7-len(idx)//5),
                fontweight="bold", color=C["dark"])

def fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140,
                bbox_inches="tight", facecolor=C["bg"])
    buf.seek(0)
    return buf

def metric_card(label, value, sub=""):
    s = (f'<div class="metric-card">'
         f'<div class="lbl">{label}</div>'
         f'<div class="val">{value}</div>')
    if sub: s += f'<div class="sub">{sub}</div>'
    return s + '</div>'

# ══════════════════════════════════════════════════════════════════════════════
#  RANK & TIE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def assign_ranks(data):
    """
    Assign ranks with averaged ranks for tied (repeated) values.
    Example: [10, 20, 20, 30]  →  ranks [1.0, 2.5, 2.5, 4.0]
    """
    data  = np.array(data, dtype=float)
    n     = len(data)
    ranks = np.empty(n)
    sidx  = np.argsort(data)
    i = 0
    while i < n:
        j = i
        while j < n-1 and data[sidx[j]] == data[sidx[j+1]]:
            j += 1
        avg_rank = (i+1 + j+1) / 2.0
        for k in range(i, j+1):
            ranks[sidx[k]] = avg_rank
        i = j+1
    return ranks

def detect_ties(data):
    vals, counts = np.unique(data, return_counts=True)
    return {float(v): int(c) for v, c in zip(vals, counts) if c > 1}

def tie_correction_total(data):
    """
    Σ (m³ - m) / 12  summed over ALL tied groups in the dataset.
    m = number of tied observations in each group.
    This is the total correction factor CF to be added to Σd².
    """
    _, counts = np.unique(data, return_counts=True)
    return sum((m**3 - m) / 12.0 for m in counts if m > 1)

def tied_groups_detail(data):
    """
    Returns list of (value, m, (m³-m)/12) for each tied group.
    Used for step-by-step display.
    """
    vals, counts = np.unique(data, return_counts=True)
    return [(float(v), int(m), (m**3 - m) / 12.0)
            for v, m in zip(vals, counts) if m > 1]

def compute_spearman(x, y):
    """
    Spearman Rank Correlation using the TEXTBOOK formula:

    Case 1 — No ties:
        R = 1 − (6 · Σd²) / (n(n²−1))

    Case 2 — Ties present:
        R = 1 − 6·[Σd² + Σ(m³−m)/12] / (n(n²−1))

    where m = number of tied observations in each tied group,
    and the correction term Σ(m³−m)/12 is summed over ALL tied
    groups across BOTH X and Y variables.

    Average ranks are assigned to tied values before computing d.
    """
    n  = len(x)
    Rx = assign_ranks(x)
    Ry = assign_ranks(y)
    d  = Rx - Ry
    d2 = d**2
    sum_d2 = d2.sum()

    ties_x = detect_ties(x)
    ties_y = detect_ties(y)
    has_ties = bool(ties_x or ties_y)

    # Correction factors per variable
    CF_x = tie_correction_total(x)   # Σ(m³-m)/12 for X
    CF_y = tie_correction_total(y)   # Σ(m³-m)/12 for Y
    CF_total = CF_x + CF_y           # Total correction added to Σd²

    # Tied group details for step-by-step display
    groups_x = tied_groups_detail(x)
    groups_y = tied_groups_detail(y)

    denom = n * (n**2 - 1)           # n(n²−1)

    if has_ties:
        # Case 2: R = 1 − 6·[Σd² + CF_total] / (n(n²−1))
        R = 1 - (6 * (sum_d2 + CF_total)) / denom
    else:
        # Case 1: R = 1 − 6·Σd² / (n(n²−1))
        R = 1 - (6 * sum_d2) / denom

    # Also compute no-tie formula for comparison display
    R_std = 1 - (6 * sum_d2) / denom

    R_abs     = abs(R)
    direction = "Positive ↑" if R >= 0 else "Negative ↓"

    return dict(
        Rx=Rx, Ry=Ry, d=d, d2=d2,
        sum_d2=sum_d2,
        CF_x=CF_x, CF_y=CF_y, CF_total=CF_total,
        groups_x=groups_x, groups_y=groups_y,
        R=R, R_std=R_std,
        R_abs=R_abs, direction=direction,
        ties_x=ties_x, ties_y=ties_y,
        has_ties=has_ties,
        n=n, denom=denom
    )

def label_r(r):
    if r >= 0.99:  return "Perfect"
    elif r >= 0.9: return "Very Strong"
    elif r >= 0.7: return "Strong"
    elif r >= 0.5: return "Moderate"
    elif r >= 0.3: return "Weak"
    else:          return "Very Weak / No"

def label_r2(r2):
    if r2 >= 0.99:   return "Excellent"
    elif r2 >= 0.95: return "Very Good"
    elif r2 >= 0.85: return "Good"
    elif r2 >= 0.70: return "Moderate"
    else:            return "Poor"

# ══════════════════════════════════════════════════════════════════════════════
#  PARSE MANUAL INPUT
# ══════════════════════════════════════════════════════════════════════════════

def parse_values(text):
    try:
        vals = [float(v.strip())
                for v in text.replace("\n", ",").split(",") if v.strip()]
        return vals, None
    except Exception:
        return None, "⚠ Invalid — use comma-separated numbers e.g.  10, 20, 30"

# ══════════════════════════════════════════════════════════════════════════════
#  GAUGE  0 to 1
# ══════════════════════════════════════════════════════════════════════════════

def draw_gauge(ax, value, title, top_label, cmap="RdYlGn"):
    ax.set_facecolor(C["panel"])
    ax.set_title(title, fontsize=10, fontweight="bold", color=C["dark"], pad=7)
    for sp in ax.spines.values(): sp.set_edgecolor(C["border"])
    ax.set_xlim(-0.08, 1.08); ax.set_ylim(-0.62, 1.62); ax.axis("off")
    grad = np.linspace(0, 1, 500).reshape(1, -1)
    ax.imshow(grad, extent=[0,1,-0.08,0.22],
              aspect="auto", cmap=cmap, vmin=0, vmax=1, zorder=1)
    ax.add_patch(plt.Rectangle((0,-0.08),1,0.30,
                 fill=False, edgecolor=C["border"], lw=1.2, zorder=2))
    ax.axvline(value, ymin=0.38, ymax=0.68, color="white", lw=3.5, zorder=5)
    ax.annotate("", xy=(value,0.30), xytext=(value,0.54),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2.2))
    for t in [0,0.25,0.5,0.75,1.0]:
        ax.text(t,-0.30,str(t),ha="center",fontsize=8,color=C["gray"])
    ax.text(0.5,-0.48,"Weak ←──────────────────→ Strong",
            ha="center",fontsize=6.5,color=C["gray"])
    ax.text(value,0.84,f"{value:.4f}",
            ha="center",fontsize=14,fontweight="bold",color="white")
    ax.text(value,1.28,top_label,
            ha="center",fontsize=8.5,fontweight="bold",color=C["green"])

# ══════════════════════════════════════════════════════════════════════════════
#  SPEARMAN PLOT  — 6 panels
# ══════════════════════════════════════════════════════════════════════════════

def plot_spearman(x, y, res, lx, ly):
    Rx,Ry = res["Rx"], res["Ry"]
    d, d2 = res["d"],  res["d2"]
    R_abs = res["R_abs"]
    n   = len(x)
    idx = np.arange(1, n+1)
    lbl = np.array([f"P{i}" for i in idx])
    bw  = max(0.3, min(0.75, 6.0/n))
    ms  = max(3, 7-n//6)

    fig = plt.figure(figsize=(17,9))
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle(
        f"Spearman Rank Correlation  |  {lx} vs {ly}  (n={n})  "
        f"|  |R| = {R_abs:.4f}  →  {label_r(R_abs)}  ({res['direction']})",
        fontsize=12, fontweight="bold", color=C["dark"], y=0.99)
    gs   = gridspec.GridSpec(2,3,hspace=0.55,wspace=0.42)
    axes = [fig.add_subplot(gs[r,c]) for r in range(2) for c in range(3)]

    # 1 — Raw scatter
    style_ax(axes[0], f"Raw Data: {lx} vs {ly}", lx, ly)
    axes[0].scatter(x, y, color=C["blue"], s=65, zorder=5,
                    edgecolors=C["bg"], lw=0.8, label="Data")
    if n <= 20:
        for i in range(n):
            axes[0].annotate(f"P{i+1}", (x[i],y[i]),
                             textcoords="offset points", xytext=(5,3),
                             fontsize=6, color=C["gray"])
    m_,b_ = np.polyfit(x,y,1)
    xl = np.linspace(x.min()-0.5, x.max()+0.5, 300)
    axes[0].plot(xl, m_*xl+b_, color=C["red"], lw=1.8, ls="--", label="Trend")
    axes[0].legend(fontsize=7, facecolor=C["panel"], labelcolor=C["dark"])

    # 2 — Rank scatter
    style_ax(axes[1], f"Rank Scatter: Rank({lx}) vs Rank({ly})",
             f"Rank({lx})", f"Rank({ly})")
    axes[1].scatter(Rx, Ry, color=C["purple"], s=65, zorder=5,
                    edgecolors=C["bg"], lw=0.8)
    if n <= 20:
        for i in range(n):
            axes[1].annotate(f"P{i+1}", (Rx[i],Ry[i]),
                             textcoords="offset points", xytext=(5,3),
                             fontsize=6, color=C["gray"])
    diag = np.linspace(1,n,200)
    axes[1].plot(diag,diag, color=C["green"], lw=1.4,
                 ls="--", alpha=0.7, label="Perfect |R|=1")
    axes[1].legend(fontsize=7, facecolor=C["panel"], labelcolor=C["dark"])

    # 3 — d bar
    style_ax(axes[2], "Rank Difference  d = Rx − Ry", "Point", "d")
    axes[2].bar(idx, d, color=[C["red"] if v<0 else C["blue"] for v in d],
                edgecolor=C["bg"], width=bw, lw=0.4)
    axes[2].axhline(0, color=C["dark"], lw=1.2)
    smart_xticks(axes[2], idx, lbl)
    bar_labels(axes[2], idx, d, fmt="{:+.1f}")

    # 4 — d² bar
    style_ax(axes[3], "Squared Differences  d²", "Point", "d²")
    axes[3].bar(idx, d2, color=C["orange"], edgecolor=C["bg"], width=bw, lw=0.4)
    if d2.mean() > 0:
        axes[3].axhline(d2.mean(), color=C["red"], lw=1.4, ls="--",
                        label=f"Mean d² = {d2.mean():.2f}")
        axes[3].legend(fontsize=7, facecolor=C["panel"], labelcolor=C["dark"])
    smart_xticks(axes[3], idx, lbl)
    bar_labels(axes[3], idx, d2, fmt="{:.1f}")

    # 5 — Rank distribution
    style_ax(axes[4], "Rank Distribution Comparison", "Point", "Rank")
    axes[4].plot(idx, Rx, "o-", color=C["blue"],   lw=1.8, ms=ms,
                 markeredgecolor=C["bg"], label=f"Rank {lx}")
    axes[4].plot(idx, Ry, "s-", color=C["purple"], lw=1.8, ms=ms,
                 markeredgecolor=C["bg"], label=f"Rank {ly}")
    axes[4].fill_between(idx, Rx, Ry, alpha=0.15, color=C["teal"])
    smart_xticks(axes[4], idx, lbl)
    axes[4].legend(fontsize=7, facecolor=C["panel"], labelcolor=C["dark"])

    # 6 — |R| gauge 0→1
    draw_gauge(axes[5], R_abs,
               "|R| Strength Gauge  (0 to 1)",
               f"{label_r(R_abs)}  ({res['direction']})")

    plt.tight_layout(rect=[0,0,1,0.97])
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  PARABOLA PLOT  — 6 panels
# ══════════════════════════════════════════════════════════════════════════════

def plot_parabola(x, y, a, b, c, y_fit, res_v, R2, lx, ly):
    n   = len(x)
    idx = np.arange(1, n+1)
    lbl = np.array([f"P{i}" for i in idx])
    bw  = max(0.3, min(0.75, 6.0/n))
    ms  = max(3, 7-n//6)
    sb  = "+" if b>=0 else "−"
    sc  = "+" if c>=0 else "−"
    eq  = f"y={a:.3f} {sb} {abs(b):.3f}x {sc} {abs(c):.4f}x²"
    xs  = np.linspace(x.min()-abs(x.max()-x.min())*0.1,
                      x.max()+abs(x.max()-x.min())*0.1, 500)
    ys  = a + b*xs + c*xs**2

    fig = plt.figure(figsize=(17,9))
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle(
        f"Parabolic Curve Fitting  (y = a + bx + cx²)  |  {lx} vs {ly}  "
        f"(n={n})  |  R² = {R2:.4f}  →  {label_r2(R2)}",
        fontsize=12, fontweight="bold", color=C["dark"], y=0.99)
    gs   = gridspec.GridSpec(2,3,hspace=0.55,wspace=0.42)
    axes = [fig.add_subplot(gs[r,c]) for r in range(2) for c in range(3)]

    # 1 — Data + parabola
    style_ax(axes[0], "Data & Fitted Parabola", lx, ly)
    axes[0].scatter(x, y, color=C["blue"], s=70, zorder=5,
                    edgecolors=C["bg"], lw=0.8, label="Observed")
    axes[0].plot(xs, ys, color=C["red"], lw=2.2, label=eq)
    axes[0].fill_between(xs, ys,
                         min(ys.min(),y.min())-abs(y.max()-y.min())*0.05,
                         alpha=0.08, color=C["red"])
    if n <= 20:
        for i in range(n):
            axes[0].annotate(f"P{i+1}", (x[i],y[i]),
                             textcoords="offset points", xytext=(5,3),
                             fontsize=6, color=C["gray"])
    axes[0].legend(fontsize=7, facecolor=C["panel"], labelcolor=C["dark"])

    # 2 — Residual bars
    style_ax(axes[1], "Residuals per Point", "Point", f"{ly} − ŷ")
    axes[1].bar(idx, res_v,
                color=[C["red"] if v<0 else C["green"] for v in res_v],
                edgecolor=C["bg"], width=bw, lw=0.4)
    axes[1].axhline(0, color=C["dark"], lw=1.2)
    smart_xticks(axes[1], idx, lbl)
    ypad = max(abs(res_v))*0.22 if max(abs(res_v))>0 else 0.5
    axes[1].set_ylim(res_v.min()-ypad, res_v.max()+ypad)
    bar_labels(axes[1], idx, res_v, fmt="{:+.2f}")

    # 3 — Actual vs Fitted scatter
    style_ax(axes[2], "Actual vs Fitted", f"Actual {ly}", "Fitted ŷ")
    axes[2].scatter(y, y_fit, color=C["purple"], s=65, zorder=5,
                    edgecolors=C["bg"], lw=0.8)
    lo_ = min(y.min(),y_fit.min())-abs(y.max()-y.min())*0.05
    hi_ = max(y.max(),y_fit.max())+abs(y.max()-y.min())*0.05
    axes[2].plot([lo_,hi_],[lo_,hi_], color=C["orange"],
                 lw=1.6, ls="--", label="Perfect fit")
    axes[2].set_xlim(lo_,hi_); axes[2].set_ylim(lo_,hi_)
    axes[2].legend(fontsize=7, facecolor=C["panel"], labelcolor=C["dark"])

    # 4 — Residual histogram
    style_ax(axes[3], "Residual Distribution", "Residual", "Frequency")
    bins = max(4, min(n, int(1+3.322*np.log10(n))))
    axes[3].hist(res_v, bins=bins, color=C["blue"],
                 edgecolor=C["bg"], lw=0.6, alpha=0.85)
    axes[3].axvline(0, color=C["red"], lw=1.6, ls="--", label="Zero")
    axes[3].axvline(res_v.mean(), color=C["yellow"], lw=1.4, ls="-.",
                    label=f"Mean = {res_v.mean():.3f}")
    axes[3].legend(fontsize=7, facecolor=C["panel"], labelcolor=C["dark"])

    # 5 — Actual vs Fitted line
    style_ax(axes[4], f"Actual vs Fitted  ({ly})", "Point", ly)
    axes[4].plot(idx, y,     "o-",  color=C["blue"], lw=1.8, ms=ms,
                 markeredgecolor=C["bg"], label="Actual")
    axes[4].plot(idx, y_fit, "s--", color=C["red"],  lw=1.8, ms=ms,
                 markeredgecolor=C["bg"], label="Fitted ŷ")
    axes[4].fill_between(idx, y, y_fit, alpha=0.15, color=C["teal"])
    smart_xticks(axes[4], idx, lbl)
    axes[4].legend(fontsize=7, facecolor=C["panel"], labelcolor=C["dark"])

    # 6 — R² gauge 0→1
    draw_gauge(axes[5], R2, "R² Goodness of Fit  (0 to 1)",
               f"{label_r2(R2)} Fit")

    plt.tight_layout(rect=[0,0,1,0.97])
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <h1>📊 EM-4 Statistical Methods</h1>
  <p>Assignment 9 &nbsp;·&nbsp; INFT Engineering &nbsp;·&nbsp;
     Spearman Rank Correlation &amp; Parabolic Curve Fitting</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs([
    "📘  Spearman Rank Correlation",
    "📗  Parabolic Curve Fitting"
])

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — SPEARMAN
# ─────────────────────────────────────────────────────────────────────────────
with tab1:

    st.markdown('<div class="section-title">📥 Choose Input Method</div>',
                unsafe_allow_html=True)

    sp_mode = st.radio("", ["📂 Upload CSV File", "✏️ Enter Values Manually"],
                       horizontal=True, key="sp_mode")

    x_sp = y_sp = lx_sp = ly_sp = None   # init

    # ── CSV Upload ────────────────────────────────────────────────────────────
    if sp_mode == "📂 Upload CSV File":
        st.markdown('<div class="section-title">📂 Upload Your CSV</div>',
                    unsafe_allow_html=True)
        st.markdown("""<div class="info-box">
            📌 <b>CSV Format:</b> First row = column headers. Each column = one variable.
            Repeated (tied) values are supported — average rank will be assigned automatically.<br>
            <b>Example CSV columns:</b> Player, Runs, Average, Strike_Rate, Wickets, Economy ...
        </div>""", unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload CSV file", type=["csv"], key="sp_csv")

        if uploaded:
            df_csv = pd.read_csv(uploaded)
            st.markdown('<div class="section-title">📋 Uploaded Data Preview</div>',
                        unsafe_allow_html=True)
            st.dataframe(df_csv, use_container_width=True, hide_index=True)

            num_cols = df_csv.select_dtypes(include=np.number).columns.tolist()
            if len(num_cols) < 2:
                st.error("⚠ CSV must have at least 2 numeric columns.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    lx_sp = st.selectbox("Select X variable (to rank)",
                                         num_cols, index=0, key="sp_cx")
                with col2:
                    ly_sp = st.selectbox("Select Y variable (to rank)",
                                         num_cols, index=1, key="sp_cy")
                x_sp = df_csv[lx_sp].dropna().values.astype(float)
                y_sp = df_csv[ly_sp].dropna().values.astype(float)

                ties_prev_x = detect_ties(x_sp)
                ties_prev_y = detect_ties(y_sp)
                if ties_prev_x or ties_prev_y:
                    msg = []
                    if ties_prev_x:
                        msg.append(f"<b>{lx_sp}</b>: " +
                            ", ".join([f"{v} appears {c} times"
                                       for v,c in ties_prev_x.items()]))
                    if ties_prev_y:
                        msg.append(f"<b>{ly_sp}</b>: " +
                            ", ".join([f"{v} appears {c} times"
                                       for v,c in ties_prev_y.items()]))
                    st.markdown(f"""<div class="tie-box">
                        ⚠️ <b>Repeated values found in selected columns:</b><br>
                        {"<br>".join(msg)}<br>
                        ✅ Average rank will be assigned. Tie correction formula will be applied automatically.
                    </div>""", unsafe_allow_html=True)

    # ── Manual Entry ──────────────────────────────────────────────────────────
    else:
        st.markdown('<div class="section-title">✏️ Enter Data Manually</div>',
                    unsafe_allow_html=True)
        st.markdown("""<div class="info-box">
            📌 Enter comma-separated values. Repeated values are handled automatically.<br>
            <b>Tip:</b> You can intentionally include repeated values to see tie handling in action.
        </div>""", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 2])
        with col1:
            n_sp = st.number_input("Number of pairs (n)", 3, 30, 10, key="n_sp")
        with col2:
            lx_sp = st.text_input("X variable name", "Maths", key="lx_sp")
        with col3:
            ly_sp = st.text_input("Y variable name", "Science", key="ly_sp")

        col4, col5 = st.columns(2)
        with col4:
            xt = st.text_area(
                f"Enter {lx_sp} values  (includes repeated values → tie handling)",
                "85, 72, 90, 60, 78, 95, 55, 88, 70, 65",
                height=100, key="x_sp_manual",
                help="Comma-separated numbers. Repeated values OK.")
        with col5:
            yt = st.text_area(
                f"Enter {ly_sp} values  (includes repeated values → tie handling)",
                "80, 68, 88, 55, 74, 92, 50, 85, 72, 60",
                height=100, key="y_sp_manual",
                help="Comma-separated numbers. Repeated values OK.")

        xv, e1 = parse_values(xt)
        yv, e2 = parse_values(yt)
        if e1: st.error(e1)
        elif e2: st.error(e2)
        elif xv and yv:
            x_sp = np.array(xv, dtype=float)
            y_sp = np.array(yv, dtype=float)

            ties_x_prev = detect_ties(x_sp)
            ties_y_prev = detect_ties(y_sp)
            if ties_x_prev or ties_y_prev:
                msg = []
                if ties_x_prev:
                    msg.append(f"<b>{lx_sp}</b>: " +
                        ", ".join([f"{v} appears {c} times"
                                   for v,c in ties_x_prev.items()]))
                if ties_y_prev:
                    msg.append(f"<b>{ly_sp}</b>: " +
                        ", ".join([f"{v} appears {c} times"
                                   for v,c in ties_y_prev.items()]))
                st.markdown(f"""<div class="tie-box">
                    ⚠️ <b>Repeated values detected:</b><br>
                    {"<br>".join(msg)}<br>
                    ✅ Average rank assigned. Tie correction Σ(m³−m)/12 will be applied.
                </div>""", unsafe_allow_html=True)

    # ── CALCULATE BUTTON ──────────────────────────────────────────────────────
    st.markdown("---")
    calc_sp = st.button("▶  Calculate Spearman Correlation", key="btn_sp")

    if calc_sp:
        if x_sp is None or y_sp is None:
            st.error("⚠ Please provide data first (upload CSV or enter values).")
        elif len(x_sp) != len(y_sp):
            st.error(f"⚠ X has {len(x_sp)} values, Y has {len(y_sp)}. Must be equal.")
        elif len(x_sp) < 3:
            st.error("⚠ Need at least 3 data pairs.")
        else:
            x = x_sp; y = y_sp
            res = compute_spearman(x, y)
            n   = len(x)

            # ── Formula display ───────────────────────────────────────────────
            if res["has_ties"]:
                # Build tie group breakdown strings
                def group_str(groups, var):
                    if not groups:
                        return f"No ties in {var}"
                    parts = []
                    for val, m, cf in groups:
                        parts.append(f"value {val} (m={m}): (m³−m)/12 = ({m}³−{m})/12 = <b>{cf:.4f}</b>")
                    return f"<b>{var}:</b> " + " &nbsp;|&nbsp; ".join(parts)

                gx_str = group_str(res["groups_x"], lx_sp)
                gy_str = group_str(res["groups_y"], ly_sp)

                st.markdown(f"""<div class="formula-box">
                    <b>📐 Case 2: Ties Present — Formula Used</b><br><br>
                    <code style="font-size:1.05rem; color:#38ef7d;">
                    R = 1 − 6·[Σd² + Σ(m³−m)/12] / n(n²−1)
                    </code><br><br>
                    <b>Step 1 — Tie Correction Factors Σ(m³−m)/12:</b><br>
                    &nbsp;&nbsp;{gx_str}<br>
                    &nbsp;&nbsp;{gy_str}<br><br>
                    <b>CF (X)</b> = {res['CF_x']:.4f} &nbsp;|&nbsp;
                    <b>CF (Y)</b> = {res['CF_y']:.4f} &nbsp;|&nbsp;
                    <b>Total CF = CF(X) + CF(Y)</b> = {res['CF_total']:.4f}<br><br>
                    <b>Step 2 — Substituting:</b><br>
                    R = 1 − 6·[{res['sum_d2']:.4f} + {res['CF_total']:.4f}] / ({n}·({n}²−1))<br>
                    R = 1 − 6·[{res['sum_d2'] + res['CF_total']:.4f}] / {res['denom']:.0f}<br>
                    R = 1 − {6*(res['sum_d2']+res['CF_total']):.4f} / {res['denom']:.0f}<br>
                    <b>R = {res['R']:.4f}</b>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="formula-box">
                    <b>📐 Case 1: No Ties — Standard Formula Used</b><br><br>
                    <code style="font-size:1.05rem; color:#38ef7d;">
                    R = 1 − 6·Σd² / n(n²−1)
                    </code><br><br>
                    <b>Substituting:</b><br>
                    R = 1 − 6·{res['sum_d2']:.4f} / ({n}·({n}²−1))<br>
                    R = 1 − {6*res['sum_d2']:.4f} / {res['denom']:.0f}<br>
                    <b>R = {res['R']:.4f}</b>
                </div>""", unsafe_allow_html=True)

            # Result badges
            st.markdown(f"""<div style="margin:16px 0;">
              <span class="result-badge">R = {res['R']:.4f} &nbsp;|&nbsp; |R| = {res['R_abs']:.4f} &nbsp;→&nbsp; {label_r(res['R_abs'])}</span>
              <span class="result-badge result-badge-orange">Direction: {res['direction']}</span>
            </div>""", unsafe_allow_html=True)

            # Metric cards
            cf_label = "Σ(m³−m)/12 (X+Y)" if res["has_ties"] else "No ties"
            cards = (
                metric_card("n (pairs)", n) +
                metric_card("Σd²", f"{res['sum_d2']:.4f}") +
                metric_card("CF_X  Σ(m³−m)/12", f"{res['CF_x']:.4f}",
                            "X correction") +
                metric_card("CF_Y  Σ(m³−m)/12", f"{res['CF_y']:.4f}",
                            "Y correction") +
                metric_card("Total CF", f"{res['CF_total']:.4f}",
                            "Added to Σd²" if res["has_ties"] else "No correction") +
                metric_card("n(n²−1)", f"{res['denom']:.0f}") +
                metric_card("|R|  (0 to 1)", f"{res['R_abs']:.4f}",
                            label_r(res['R_abs'])) +
                metric_card("Direction", res['direction'])
            )
            st.markdown(f'<div class="metric-row">{cards}</div>',
                        unsafe_allow_html=True)

            st.markdown("""<div class="info-box">
                📌 <b>Why |R| (0 to 1)?</b>&nbsp; Raw R ranges from −1 to +1.
                We show |R| (absolute value) so the gauge always reads 0 = no correlation,
                1 = perfect correlation. The <b>Direction</b> badge tells you whether
                the relationship is Positive ↑ or Negative ↓.
            </div>""", unsafe_allow_html=True)

            # Working Table
            st.markdown('<div class="section-title">📋 Working Table</div>',
                        unsafe_allow_html=True)
            df_wt = pd.DataFrame({
                "i"             : range(1, n+1),
                lx_sp           : x,
                ly_sp           : y,
                f"Rank({lx_sp})": res["Rx"],
                f"Rank({ly_sp})": res["Ry"],
                "d = Rx − Ry"  : res["d"],
                "d²"            : res["d2"],
            })
            total_row = pd.DataFrame([{
                "i": "Σ", lx_sp:"—", ly_sp:"—",
                f"Rank({lx_sp})":"—", f"Rank({ly_sp})":"—",
                "d = Rx − Ry":"—",
                "d²": f"{res['sum_d2']:.2f}"
            }])
            st.dataframe(pd.concat([df_wt, total_row], ignore_index=True),
                         use_container_width=True, hide_index=True)

            # Graphs
            st.markdown('<div class="section-title">📈 Visual Analysis (6 Graphs)</div>',
                        unsafe_allow_html=True)
            fig = plot_spearman(x, y, res, lx_sp, ly_sp)
            st.pyplot(fig, use_container_width=True)
            st.download_button("⬇  Download Graph as PNG", fig_to_buf(fig),
                               "spearman_analysis.png", "image/png")
            plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — PARABOLA
# ─────────────────────────────────────────────────────────────────────────────
with tab2:

    st.markdown('<div class="section-title">📥 Choose Input Method</div>',
                unsafe_allow_html=True)

    pb_mode = st.radio("", ["📂 Upload CSV File", "✏️ Enter Values Manually"],
                       horizontal=True, key="pb_mode")

    x_pb = y_pb = lx_pb = ly_pb = None

    # ── CSV Upload ────────────────────────────────────────────────────────────
    if pb_mode == "📂 Upload CSV File":
        st.markdown('<div class="section-title">📂 Upload Your CSV</div>',
                    unsafe_allow_html=True)
        st.markdown("""<div class="info-box">
            📌 <b>CSV Format:</b> First row = column headers. Select which column is X and which is Y.<br>
            The parabola y = a + bx + cx² will be fitted to your selected columns.
        </div>""", unsafe_allow_html=True)

        uploaded_pb = st.file_uploader("Upload CSV file", type=["csv"], key="pb_csv")

        if uploaded_pb:
            df_pb_csv = pd.read_csv(uploaded_pb)
            st.markdown('<div class="section-title">📋 Uploaded Data Preview</div>',
                        unsafe_allow_html=True)
            st.dataframe(df_pb_csv, use_container_width=True, hide_index=True)

            num_cols_pb = df_pb_csv.select_dtypes(include=np.number).columns.tolist()
            if len(num_cols_pb) < 2:
                st.error("⚠ CSV must have at least 2 numeric columns.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    lx_pb = st.selectbox("Select X variable", num_cols_pb,
                                         index=0, key="pb_cx")
                with col2:
                    ly_pb = st.selectbox("Select Y variable", num_cols_pb,
                                         index=1, key="pb_cy")
                x_pb = df_pb_csv[lx_pb].dropna().values.astype(float)
                y_pb = df_pb_csv[ly_pb].dropna().values.astype(float)

    # ── Manual Entry ──────────────────────────────────────────────────────────
    else:
        st.markdown('<div class="section-title">✏️ Enter Data Manually</div>',
                    unsafe_allow_html=True)
        st.markdown("""<div class="info-box">
            📌 Enter x and y values as comma-separated numbers.<br>
            The best-fit parabola <b>y = a + bx + cx²</b> will be calculated using
            the Least Squares Method (Normal Equations → Matrix Solve).
        </div>""", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 2])
        with col1:
            n_pb = st.number_input("Number of points (n)", 3, 30, 9, key="n_pb")
        with col2:
            lx_pb = st.text_input("X variable name", "Speed", key="lx_pb")
        with col3:
            ly_pb = st.text_input("Y variable name", "Fuel",  key="ly_pb")

        col4, col5 = st.columns(2)
        with col4:
            xpt = st.text_area(f"Enter {lx_pb} values",
                               "20, 30, 40, 50, 60, 70, 80, 90, 100",
                               height=100, key="x_pb_manual")
        with col5:
            ypt = st.text_area(f"Enter {ly_pb} values",
                               "12, 16, 20, 24, 26, 25, 22, 18, 13",
                               height=100, key="y_pb_manual")

        xv2, e1 = parse_values(xpt)
        yv2, e2 = parse_values(ypt)
        if e1: st.error(e1)
        elif e2: st.error(e2)
        elif xv2 and yv2:
            x_pb = np.array(xv2, dtype=float)
            y_pb = np.array(yv2, dtype=float)

    # ── CALCULATE BUTTON ──────────────────────────────────────────────────────
    st.markdown("---")
    calc_pb = st.button("▶  Calculate Parabolic Fit", key="btn_pb")

    if calc_pb:
        if x_pb is None or y_pb is None:
            st.error("⚠ Please provide data first (upload CSV or enter values).")
        elif len(x_pb) != len(y_pb):
            st.error(f"⚠ X has {len(x_pb)} values, Y has {len(y_pb)}. Must be equal.")
        elif len(x_pb) < 3:
            st.error("⚠ Need at least 3 data points.")
        else:
            x = x_pb; y = y_pb; n = len(x)
            x2=x**2; x3=x**3; x4=x**4; xy=x*y; x2y=x2*y

            sx,sx2,sx3,sx4 = x.sum(),x2.sum(),x3.sum(),x4.sum()
            sy,sxy,sx2y_   = y.sum(),xy.sum(),x2y.sum()

            A = np.array([[n,  sx,  sx2],
                          [sx, sx2, sx3],
                          [sx2,sx3, sx4]])
            B = np.array([sy, sxy, sx2y_])
            a, b, c = np.linalg.solve(A, B)

            y_fit  = a + b*x + c*x2
            res_v  = y - y_fit
            ss_res = (res_v**2).sum()
            ss_tot = ((y-y.mean())**2).sum()
            R2     = 1 - ss_res/ss_tot if ss_tot != 0 else 1.0
            sb = "+" if b>=0 else "−"
            sc = "+" if c>=0 else "−"

            # Result badges
            st.markdown(f"""<div style="margin:16px 0;">
              <span class="result-badge result-badge-blue">
                y = {a:.4f} {sb} {abs(b):.4f}x {sc} {abs(c):.6f}x²
              </span>
              <span class="result-badge">
                R² = {R2:.4f} &nbsp;→&nbsp; {label_r2(R2)}
              </span>
            </div>""", unsafe_allow_html=True)

            # Metric cards
            cards = (
                metric_card("n (points)", n) +
                metric_card("a (constant)",   f"{a:.5f}") +
                metric_card("b (linear)",     f"{b:.5f}") +
                metric_card("c (quadratic)",  f"{c:.6f}") +
                metric_card("R²  (0 to 1)",   f"{R2:.4f}", label_r2(R2))
            )
            st.markdown(f'<div class="metric-row">{cards}</div>',
                        unsafe_allow_html=True)

            # Working table
            st.markdown('<div class="section-title">📋 Working Table</div>',
                        unsafe_allow_html=True)
            df_w = pd.DataFrame({
                "i":range(1,n+1), lx_pb:x, ly_pb:y,
                "x²":x2,"x³":x3,"x⁴":x4,"xy":xy,"x²y":x2y
            })
            tot = pd.DataFrame([{"i":"Σ",lx_pb:sx,ly_pb:sy,
                                  "x²":sx2,"x³":sx3,"x⁴":sx4,
                                  "xy":sxy,"x²y":sx2y_}])
            st.dataframe(pd.concat([df_w,tot],ignore_index=True),
                         use_container_width=True, hide_index=True)

            # Normal equations
            st.markdown('<div class="section-title">📐 Normal Equations</div>',
                        unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "Equation":["Eq 1","Eq 2","Eq 3"],
                "LHS":[f"{sy:.3f}",f"{sxy:.3f}",f"{sx2y_:.3f}"],
                "RHS":[
                    f"= {n}·a + {sx:.3f}·b + {sx2:.3f}·c",
                    f"= {sx:.3f}·a + {sx2:.3f}·b + {sx3:.3f}·c",
                    f"= {sx2:.3f}·a + {sx3:.3f}·b + {sx4:.3f}·c",
                ]
            }), use_container_width=True, hide_index=True)

            # Fitted vs Actual
            st.markdown('<div class="section-title">📋 Fitted vs Actual Values</div>',
                        unsafe_allow_html=True)
            df_f = pd.DataFrame({
                "i":range(1,n+1), lx_pb:x, f"{ly_pb} actual":y,
                "ŷ fitted":np.round(y_fit,4),
                "Residual y−ŷ":np.round(res_v,4),
                "(Residual)²":np.round(res_v**2,6),
            })
            tot_f = pd.DataFrame([{
                "i":"Σ",lx_pb:"—",f"{ly_pb} actual":"—",
                "ŷ fitted":"—","Residual y−ŷ":"Σ(y−ŷ)² =",
                "(Residual)²":round(ss_res,6)
            }])
            st.dataframe(pd.concat([df_f,tot_f],ignore_index=True),
                         use_container_width=True, hide_index=True)

            # Graphs
            st.markdown('<div class="section-title">📈 Visual Analysis (6 Graphs)</div>',
                        unsafe_allow_html=True)
            fig = plot_parabola(x,y,a,b,c,y_fit,res_v,R2,lx_pb,ly_pb)
            st.pyplot(fig, use_container_width=True)
            st.download_button("⬇  Download Graph as PNG", fig_to_buf(fig),
                               "parabola_analysis.png", "image/png")
            plt.close(fig)

# Footer
st.markdown("""
<div style="text-align:center;color:rgba(255,255,255,0.22);
            font-size:0.76rem;margin-top:48px;padding:16px;">
  Assignment 9 &nbsp;·&nbsp; EM-4 (BSC07) &nbsp;·&nbsp; INFT Engineering
</div>""", unsafe_allow_html=True)