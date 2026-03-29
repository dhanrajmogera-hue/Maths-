"""
=============================================================================
app.py
Member 2 : [Dhanraj Mogera] — Streamlit UI, Tabs, Input Handling
=============================================================================
Commits to make:
  Commit 1: "Added Streamlit UI with two tabs and CSS styling"
  Commit 2: "Added CSV upload and manual input handling for both methods"
=============================================================================
Run: streamlit run app.py
=============================================================================
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Import from team members' files
from utils     import parse_values, metric_card, fig_to_buf, label_r, label_r2
from spearman  import compute_spearman, detect_ties, plot_spearman
from parabola  import compute_parabola, plot_parabola

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
#  CSS STYLING
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
.main-header h1 { font-family:'Space Mono',monospace; color:white;
    font-size:1.8rem; margin:0 0 6px 0; }
.main-header p { color:rgba(255,255,255,0.72); font-size:0.90rem; margin:0; }

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
.metric-card .lbl { color:rgba(255,255,255,0.48); font-size:0.68rem;
    text-transform:uppercase; letter-spacing:1px; margin-bottom:4px; }
.metric-card .val { color:white; font-family:'Space Mono',monospace;
    font-size:1.1rem; font-weight:700; }
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
.stTextInput>label,.stNumberInput>label,.stSelectbox>label {
    color:rgba(255,255,255,0.8) !important; font-weight:500; }
.stTextInput input,.stNumberInput input {
    background:rgba(255,255,255,0.07) !important;
    border:1px solid rgba(255,255,255,0.14) !important;
    color:white !important; border-radius:8px !important; }
.stTextArea textarea {
    background:rgba(255,255,255,0.07) !important;
    border:1px solid rgba(255,255,255,0.14) !important;
    color:white !important; border-radius:8px !important;
    font-family:'Space Mono',monospace !important; font-size:0.85rem !important; }
.stButton>button {
    background:linear-gradient(90deg,#667eea,#764ba2) !important;
    color:white !important; border:none !important; border-radius:10px !important;
    font-weight:600 !important; font-size:1rem !important;
    padding:10px 28px !important; width:100%;
    box-shadow:0 4px 14px rgba(102,126,234,0.4) !important; }
.stTabs [data-baseweb="tab"]   { color:rgba(255,255,255,0.5) !important; }
.stTabs [aria-selected="true"] { color:white !important; font-weight:700; }
[data-testid="stFileUploader"] label { color:rgba(255,255,255,0.8) !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <h1>📊 EM-4 Statistical Methods</h1>
  <p>Assignment 9 &nbsp;·&nbsp; INFT Semester 4 &nbsp;·&nbsp;
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

    x_sp = y_sp = lx_sp = ly_sp = None

    # ── CSV Upload ─────────────────────────────────────────────────────────
    if sp_mode == "📂 Upload CSV File":
        st.markdown("""<div class="info-box">
            📌 <b>CSV Format:</b> First row = column headers. Each column = one variable.<br>
            Repeated (tied) values are supported — average rank assigned automatically.<br>
            💡 Try uploading <b>ipl_batsmen_100.csv</b> and select Runs vs Average for strong correlation.
        </div>""", unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="sp_csv")
        if uploaded:
            df_csv = pd.read_csv(uploaded)
            st.markdown('<div class="section-title">📋 Data Preview</div>',
                        unsafe_allow_html=True)
            st.dataframe(df_csv, use_container_width=True, hide_index=True)

            num_cols = df_csv.select_dtypes(include=np.number).columns.tolist()
            if len(num_cols) < 2:
                st.error("⚠ CSV needs at least 2 numeric columns.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    lx_sp = st.selectbox("X variable", num_cols, index=0, key="sp_cx")
                with col2:
                    ly_sp = st.selectbox("Y variable", num_cols, index=1, key="sp_cy")
                x_sp = df_csv[lx_sp].dropna().values.astype(float)
                y_sp = df_csv[ly_sp].dropna().values.astype(float)

                # Live tie preview
                tx = detect_ties(x_sp); ty = detect_ties(y_sp)
                if tx or ty:
                    msg = []
                    if tx: msg.append(f"<b>{lx_sp}</b>: " + ", ".join([f"{v} appears {c}×" for v,c in tx.items()]))
                    if ty: msg.append(f"<b>{ly_sp}</b>: " + ", ".join([f"{v} appears {c}×" for v,c in ty.items()]))
                    st.markdown(f"""<div class="tie-box">
                        ⚠️ <b>Repeated values found:</b><br>{"<br>".join(msg)}<br>
                        ✅ Average rank will be assigned automatically.
                    </div>""", unsafe_allow_html=True)

    # ── Manual Entry ───────────────────────────────────────────────────────
    else:
        st.markdown("""<div class="info-box">
            📌 Enter comma-separated values. Repeated values handled automatically.<br>
            💡 Try entering values with repeats to see tie handling in action.
        </div>""", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 2])
        with col1: n_sp  = st.number_input("n (pairs)", 3, 30, 10, key="n_sp")
        with col2: lx_sp = st.text_input("X variable name", "Maths",   key="lx_sp")
        with col3: ly_sp = st.text_input("Y variable name", "Science", key="ly_sp")

        col4, col5 = st.columns(2)
        with col4:
            xt = st.text_area(f"{lx_sp} values",
                              "85,72,90,60,78,95,55,88,70,65",
                              height=100, key="x_sp_m")
        with col5:
            yt = st.text_area(f"{ly_sp} values",
                              "80,68,88,55,74,92,50,85,72,60",
                              height=100, key="y_sp_m")

        xv, e1 = parse_values(xt)
        yv, e2 = parse_values(yt)
        if e1: st.error(e1)
        elif e2: st.error(e2)
        elif xv and yv:
            x_sp = np.array(xv, dtype=float)
            y_sp = np.array(yv, dtype=float)
            tx = detect_ties(x_sp); ty = detect_ties(y_sp)
            if tx or ty:
                msg = []
                if tx: msg.append(f"<b>{lx_sp}</b>: " + ", ".join([f"{v} (×{c})" for v,c in tx.items()]))
                if ty: msg.append(f"<b>{ly_sp}</b>: " + ", ".join([f"{v} (×{c})" for v,c in ty.items()]))
                st.markdown(f"""<div class="tie-box">
                    ⚠️ <b>Repeated values detected:</b><br>{"<br>".join(msg)}<br>
                    ✅ Correction factor T applied automatically.
                </div>""", unsafe_allow_html=True)

    # ── Calculate ──────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("▶  Calculate Spearman Correlation", key="btn_sp"):
        if x_sp is None or y_sp is None:
            st.error("⚠ Provide data first.")
        elif len(x_sp) != len(y_sp):
            st.error(f"⚠ X has {len(x_sp)} values, Y has {len(y_sp)}. Must match.")
        elif len(x_sp) < 3:
            st.error("⚠ Need at least 3 pairs.")
        else:
            x = x_sp; y = y_sp
            res = compute_spearman(x, y)
            n   = len(x)

            if res["has_ties"]:
                msg = []
                if res["ties_x"]: msg.append(f"<b>{lx_sp}</b>: " + ", ".join([f"{v} (×{c})" for v,c in res["ties_x"].items()]))
                if res["ties_y"]: msg.append(f"<b>{ly_sp}</b>: " + ", ".join([f"{v} (×{c})" for v,c in res["ties_y"].items()]))
                st.markdown(f"""<div class="tie-box">
                    ⚠️ <b>Tied values — Correction Applied</b><br>{"<br>".join(msg)}<br>
                    Tx={res['Tx']:.4f} | Ty={res['Ty']:.4f} | Ax={res['Ax']:.4f} | Ay={res['Ay']:.4f}<br>
                    Formula: R = (Ax + Ay − Σd²) / (2√(Ax·Ay)) = <b>{res['R_corr']:.4f}</b>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""<div style="margin:16px 0;">
              <span class="result-badge">|R| = {res['R_abs']:.4f} &nbsp;→&nbsp; {label_r(res['R_abs'])}</span>
              <span class="result-badge result-badge-orange">Direction: {res['direction']}</span>
            </div>""", unsafe_allow_html=True)

            cards = (
                metric_card("n (pairs)", n) +
                metric_card("Σd²", f"{res['d2'].sum():.4f}") +
                metric_card("Tx", f"{res['Tx']:.4f}") +
                metric_card("Ty", f"{res['Ty']:.4f}") +
                metric_card("R Standard", f"{res['R_std']:.4f}", "No tie correction") +
                metric_card("R Corrected", f"{res['R_corr']:.4f}", "With tie correction") +
                metric_card("|R| (0–1)", f"{res['R_abs']:.4f}", label_r(res["R_abs"])) +
                metric_card("Direction", res["direction"])
            )
            st.markdown(f'<div class="metric-row">{cards}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">📋 Working Table</div>', unsafe_allow_html=True)
            df_wt = pd.DataFrame({
                "i": range(1, n+1), lx_sp: x, ly_sp: y,
                f"Rank({lx_sp})": res["Rx"], f"Rank({ly_sp})": res["Ry"],
                "d = Rx−Ry": res["d"], "d²": res["d2"],
            })
            tot = pd.DataFrame([{"i":"Σ", lx_sp:"—", ly_sp:"—",
                                  f"Rank({lx_sp})":"—", f"Rank({ly_sp})":"—",
                                  "d = Rx−Ry":"—", "d²": f"{res['d2'].sum():.2f}"}])
            st.dataframe(pd.concat([df_wt, tot], ignore_index=True),
                         use_container_width=True, hide_index=True)

            st.markdown('<div class="section-title">📈 Visual Analysis (6 Graphs)</div>',
                        unsafe_allow_html=True)
            fig = plot_spearman(x, y, res, lx_sp, ly_sp)
            st.pyplot(fig, use_container_width=True)
            st.download_button("⬇  Download Graph", fig_to_buf(fig),
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

    if pb_mode == "📂 Upload CSV File":
        st.markdown("""<div class="info-box">
            📌 Upload any CSV. Select X and Y numeric columns.<br>
            The best-fit parabola  <b>y = a + bx + cx²</b>  will be calculated.
        </div>""", unsafe_allow_html=True)

        uploaded_pb = st.file_uploader("Upload CSV", type=["csv"], key="pb_csv")
        if uploaded_pb:
            df_pb = pd.read_csv(uploaded_pb)
            st.dataframe(df_pb, use_container_width=True, hide_index=True)
            num_cols_pb = df_pb.select_dtypes(include=np.number).columns.tolist()
            if len(num_cols_pb) < 2:
                st.error("⚠ CSV needs at least 2 numeric columns.")
            else:
                col1, col2 = st.columns(2)
                with col1: lx_pb = st.selectbox("X variable", num_cols_pb, index=0, key="pb_cx")
                with col2: ly_pb = st.selectbox("Y variable", num_cols_pb, index=1, key="pb_cy")
                x_pb = df_pb[lx_pb].dropna().values.astype(float)
                y_pb = df_pb[ly_pb].dropna().values.astype(float)
    else:
        st.markdown("""<div class="info-box">
            📌 Enter x and y values. The parabola  <b>y = a + bx + cx²</b>  will be fitted
            using the Least Squares Method (Normal Equations → Matrix Solve).
        </div>""", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 2])
        with col1: n_pb  = st.number_input("n (points)", 3, 30, 9,       key="n_pb")
        with col2: lx_pb = st.text_input("X variable name", "Speed",    key="lx_pb")
        with col3: ly_pb = st.text_input("Y variable name", "Fuel",     key="ly_pb")

        col4, col5 = st.columns(2)
        with col4: xpt = st.text_area(f"{lx_pb} values", "20,30,40,50,60,70,80,90,100", height=100, key="x_pb_m")
        with col5: ypt = st.text_area(f"{ly_pb} values", "12,16,20,24,26,25,22,18,13",  height=100, key="y_pb_m")

        xv2, e1 = parse_values(xpt)
        yv2, e2 = parse_values(ypt)
        if e1: st.error(e1)
        elif e2: st.error(e2)
        elif xv2 and yv2:
            x_pb = np.array(xv2, dtype=float)
            y_pb = np.array(yv2, dtype=float)

    st.markdown("---")
    if st.button("▶  Calculate Parabolic Fit", key="btn_pb"):
        if x_pb is None or y_pb is None:
            st.error("⚠ Provide data first.")
        elif len(x_pb) != len(y_pb):
            st.error(f"⚠ X={len(x_pb)} values, Y={len(y_pb)}. Must match.")
        elif len(x_pb) < 3:
            st.error("⚠ Need at least 3 points.")
        else:
            x = x_pb; y = y_pb
            res = compute_parabola(x, y)
            a, b, c = res["a"], res["b"], res["c"]
            R2 = res["R2"]; t = res["table"]; n = res["n"]
            sb = "+" if b>=0 else "−"; sc = "+" if c>=0 else "−"

            st.markdown(f"""<div style="margin:16px 0;">
              <span class="result-badge result-badge-blue">
                y = {a:.4f} {sb} {abs(b):.4f}x {sc} {abs(c):.6f}x²
              </span>
              <span class="result-badge">R² = {R2:.4f} → {label_r2(R2)}</span>
            </div>""", unsafe_allow_html=True)

            cards = (
                metric_card("n (points)", n) +
                metric_card("a (constant)",  f"{a:.5f}") +
                metric_card("b (linear)",    f"{b:.5f}") +
                metric_card("c (quadratic)", f"{c:.6f}") +
                metric_card("R² (0–1)",      f"{R2:.4f}", label_r2(R2))
            )
            st.markdown(f'<div class="metric-row">{cards}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">📋 Working Table</div>', unsafe_allow_html=True)
            df_w = pd.DataFrame({"i":range(1,n+1), lx_pb:x, ly_pb:y,
                                  "x²":t["x2"],"x³":t["x3"],"x⁴":t["x4"],
                                  "xy":t["xy"],"x²y":t["x2y"]})
            tot_w = pd.DataFrame([{"i":"Σ",lx_pb:t["sx"],ly_pb:t["sy"],
                                    "x²":t["sx2"],"x³":t["sx3"],"x⁴":t["sx4"],
                                    "xy":t["sxy"],"x²y":t["sx2y"]}])
            st.dataframe(pd.concat([df_w,tot_w],ignore_index=True),
                         use_container_width=True, hide_index=True)

            st.markdown('<div class="section-title">📐 Normal Equations</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "Eq":["Eq 1","Eq 2","Eq 3"],
                "LHS":[f"{t['sy']:.3f}",f"{t['sxy']:.3f}",f"{t['sx2y']:.3f}"],
                "RHS":[
                    f"= {n}·a + {t['sx']:.3f}·b + {t['sx2']:.3f}·c",
                    f"= {t['sx']:.3f}·a + {t['sx2']:.3f}·b + {t['sx3']:.3f}·c",
                    f"= {t['sx2']:.3f}·a + {t['sx3']:.3f}·b + {t['sx4']:.3f}·c",
                ]
            }), use_container_width=True, hide_index=True)

            st.markdown('<div class="section-title">📋 Fitted vs Actual</div>', unsafe_allow_html=True)
            df_f = pd.DataFrame({"i":range(1,n+1), lx_pb:x,
                                  f"{ly_pb} actual":y,
                                  "ŷ fitted":np.round(res["y_fit"],4),
                                  "Residual y−ŷ":np.round(res["res_v"],4),
                                  "(Residual)²":np.round(res["res_v"]**2,6)})
            tot_f = pd.DataFrame([{"i":"Σ",lx_pb:"—",f"{ly_pb} actual":"—",
                                    "ŷ fitted":"—","Residual y−ŷ":"Σ(y−ŷ)²=",
                                    "(Residual)²":round(res["ss_res"],6)}])
            st.dataframe(pd.concat([df_f,tot_f],ignore_index=True),
                         use_container_width=True, hide_index=True)

            st.markdown('<div class="section-title">📈 Visual Analysis (6 Graphs)</div>',
                        unsafe_allow_html=True)
            fig = plot_parabola(x, y, res, lx_pb, ly_pb)
            st.pyplot(fig, use_container_width=True)
            st.download_button("⬇  Download Graph", fig_to_buf(fig),
                               "parabola_analysis.png", "image/png")
            plt.close(fig)

# Footer
st.markdown("""
<div style="text-align:center;color:rgba(255,255,255,0.22);
            font-size:0.76rem;margin-top:48px;padding:16px;">
  EM-4 (BSC07) &nbsp;·&nbsp; INFT Semester 4 &nbsp;·&nbsp; Mini Project
</div>""", unsafe_allow_html=True)
