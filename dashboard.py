"""
dashboard.py  —  Modern professional UI
Only this file changes. All logic stays in main.py / db.py / trainer.py / preprocessor.py.
"""

import json
import streamlit as st
import pandas as pd

from db import get_db_connection, load_books_from_db, load_latest_model, init_db
from main import recommend_books
from trainer import train_all_models
from preprocessor import FEATURE_VERSION

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LibraryIQ — Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #080d1a !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stAppViewContainer"] { background: #080d1a !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stMain"] { background: #080d1a !important; }
section[data-testid="stSidebar"] {
    background: #0d1425 !important;
    border-right: 1px solid #1e2d4a !important;
}
section[data-testid="stSidebar"] * { color: #c8d0e0 !important; }

/* ── Typography ── */
h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0a1628 0%, #0d2040 40%, #091830 100%);
    border: 1px solid #1a3060;
    border-radius: 20px;
    padding: 48px 52px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 20%;
    width: 400px; height: 200px;
    background: radial-gradient(circle, rgba(99,102,241,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #818cf8, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 10px 0;
    line-height: 1.1;
}
.hero-sub {
    color: #7a8aaa;
    font-size: 1.05rem;
    font-weight: 300;
    margin: 0;
    letter-spacing: 0.02em;
}
.hero-badge {
    display: inline-block;
    background: rgba(59,130,246,0.12);
    border: 1px solid rgba(59,130,246,0.3);
    color: #60a5fa;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 16px;
}

/* ── Section header ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: -0.01em;
    margin: 36px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1e3a5f, transparent);
    margin-left: 8px;
}

/* ── Stat cards ── */
.stat-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin-bottom: 28px; }
.stat-card {
    background: linear-gradient(135deg, #0d1e36, #0a1628);
    border: 1px solid #1a3060;
    border-radius: 14px;
    padding: 22px 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: #3b82f6; }
.stat-number {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.stat-label { color: #64748b; font-size: 0.78rem; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 6px; }

/* ── Panel / glass card ── */
.panel {
    background: #0d1425;
    border: 1px solid #1a2d4a;
    border-radius: 16px;
    padding: 24px 26px;
    margin-bottom: 20px;
}

/* ── Model pills ── */
.model-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }
.model-pill {
    background: #0f1e38;
    border: 1px solid #1e3a5f;
    border-radius: 50px;
    padding: 8px 18px;
    font-size: 0.82rem;
    color: #94a3b8;
    display: flex; align-items: center; gap: 8px;
}
.model-pill.best {
    background: linear-gradient(135deg, #1e3a6e, #162d58);
    border-color: #3b82f6;
    color: #93c5fd;
}
.model-pill .dot { width: 7px; height: 7px; border-radius: 50%; background: #3b82f6; }
.model-pill.best .dot { background: #22d3ee; box-shadow: 0 0 6px #22d3ee; }

/* ── Metric chips ── */
.metric-row { display: flex; gap: 10px; flex-wrap: wrap; }
.metric-chip {
    background: #0a1628;
    border: 1px solid #1a3060;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 0.78rem;
    color: #64748b;
    text-align: center;
    min-width: 80px;
}
.metric-chip .val { font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; color: #60a5fa; display: block; margin-bottom: 2px; }

/* ── Form area ── */
.form-panel {
    background: linear-gradient(135deg, #0d1e38, #091628);
    border: 1px solid #1e3a5f;
    border-radius: 18px;
    padding: 32px;
    margin-bottom: 24px;
}

/* ── Streamlit widget overrides ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stSlider"] label,
div[data-testid="stCheckbox"] label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stNumberInput"] input {
    background: #0a1628 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
div[data-testid="stSelectbox"] > div > div:focus-within,
div[data-testid="stNumberInput"] input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
}

/* ── Submit button ── */
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #2563eb, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 36px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    width: 100% !important;
    transition: opacity 0.2s, transform 0.1s !important;
    cursor: pointer !important;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

/* ── Sidebar buttons ── */
section[data-testid="stSidebar"] button {
    background: #0f1e38 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 10px !important;
    color: #93c5fd !important;
    width: 100% !important;
    font-size: 0.82rem !important;
    margin-bottom: 6px !important;
}
section[data-testid="stSidebar"] button:hover {
    border-color: #3b82f6 !important;
    background: #162d58 !important;
}

/* ── Rec card ── */
.rec-card {
    background: linear-gradient(135deg, #0d1e38, #0a1830);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 24px 26px;
    margin-bottom: 14px;
    transition: border-color 0.2s, transform 0.15s;
    position: relative;
    overflow: hidden;
}
.rec-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #3b82f6, #6366f1, #22d3ee);
}
.rec-card:hover { border-color: #3b82f6; transform: translateY(-2px); }
.rec-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 0 0 6px 0;
}
.rec-course { color: #60a5fa; font-size: 0.8rem; font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 14px; }
.rec-tags { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }
.tag {
    background: #0a1628;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.74rem;
    color: #94a3b8;
}
.rec-score-bar { margin: 10px 0; }
.bar-label { display: flex; justify-content: space-between; font-size: 0.74rem; color: #64748b; margin-bottom: 4px; }
.bar-track { background: #0a1628; border-radius: 4px; height: 5px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #3b82f6, #6366f1); }
.confidence-badge {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 0.74rem; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; padding: 4px 10px; border-radius: 20px;
}
.conf-high   { background: rgba(34,197,94,0.12);  border: 1px solid rgba(34,197,94,0.3);  color: #4ade80; }
.conf-medium { background: rgba(234,179,8,0.12);  border: 1px solid rgba(234,179,8,0.3);  color: #facc15; }
.conf-low    { background: rgba(239,68,68,0.12);  border: 1px solid rgba(239,68,68,0.3);  color: #f87171; }
.rec-explanation {
    background: rgba(59,130,246,0.06);
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 0.82rem;
    color: #7a8aaa;
    margin-top: 12px;
    font-style: italic;
}
.rec-rank {
    position: absolute;
    top: 18px; right: 20px;
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: rgba(59,130,246,0.12);
}

/* ── Table override ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1a3060 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] table { background: #0a1628 !important; }
[data-testid="stDataFrame"] th { background: #0d1e38 !important; color: #60a5fa !important; font-family: 'Syne', sans-serif !important; font-size: 0.75rem !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; }
[data-testid="stDataFrame"] td { color: #94a3b8 !important; font-size: 0.82rem !important; border-color: #1a2d4a !important; }

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 12px !important; border: 1px solid #1e3a5f !important; background: #0d1e38 !important; }

/* ── Divider ── */
hr { border-color: #1a2d4a !important; margin: 32px 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #080d1a; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 20px 0;'>
        <div style='font-family: Syne, sans-serif; font-size: 1.3rem; font-weight: 800;
                    background: linear-gradient(90deg,#60a5fa,#818cf8);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text; margin-bottom: 4px;'>LibraryIQ</div>
        <div style='color: #64748b; font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase;'>Admin Panel</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='color:#475569; font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:8px;'>Database</div>", unsafe_allow_html=True)
    if st.button("⚡  Initialize DB"):
        init_db()
        st.success("DB initialized.")

    st.markdown("<div style='color:#475569; font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase; margin: 16px 0 8px;'>Models</div>", unsafe_allow_html=True)
    if st.button("🤖  Train / Retrain Models"):
        with st.spinner("Training all 3 models..."):
            train_all_models()
        st.success("Training complete.")

    if st.button("🔄  Reload Page"):
        st.rerun()

    st.markdown("<hr style='border-color:#1a2d4a; margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:#475569; font-size:0.72rem;'>Feature version: <span style='color:#60a5fa;'>{FEATURE_VERSION}</span></div>", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
conn = get_db_connection()
books_df = load_books_from_db(conn)

if books_df.empty:
    st.error("No books in the database. Open the sidebar and click **Initialize DB**.")
    st.stop()

model_obj, feature_cols, model_metrics = load_latest_model(conn)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">📚 AI-Powered Recommender</div>
    <div class="hero-title">Library Intelligence</div>
    <p class="hero-sub">Discover the right books for your course, semester, and study timeline — powered by machine learning.</p>
</div>
""", unsafe_allow_html=True)

# ── Stat cards ────────────────────────────────────────────────────────────────
total_books    = len(books_df)
total_courses  = books_df["course"].nunique()
highly_rec     = len(books_df[books_df["label"] == "Highly Recommended"])
avg_score      = int(books_df["past_usage_score"].mean())

st.markdown(f"""
<div class="stat-grid">
    <div class="stat-card">
        <div class="stat-number">{total_books}</div>
        <div class="stat-label">Total Books</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">{total_courses}</div>
        <div class="stat-label">Courses</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">{highly_rec}</div>
        <div class="stat-label">Highly Recommended</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">{avg_score}</div>
        <div class="stat-label">Avg Usage Score</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Dataset + Charts ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("<div style='color:#60a5fa; font-size:0.75rem; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:12px;'>Book Catalogue (sample)</div>", unsafe_allow_html=True)
    st.dataframe(
        books_df[["title", "course", "semester", "difficulty", "past_usage_score", "label"]].head(10),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    # Course breakdown
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("<div style='color:#60a5fa; font-size:0.75rem; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:14px;'>Books per Course</div>", unsafe_allow_html=True)
    course_counts = books_df["course"].value_counts()
    max_count = course_counts.max()
    for course, count in course_counts.items():
        pct = int((count / max_count) * 100)
        st.markdown(f"""
        <div style='margin-bottom:10px;'>
            <div style='display:flex; justify-content:space-between; font-size:0.78rem; color:#94a3b8; margin-bottom:4px;'>
                <span>{course}</span><span style='color:#60a5fa; font-weight:600;'>{count}</span>
            </div>
            <div class='bar-track'><div class='bar-fill' style='width:{pct}%;'></div></div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Label breakdown
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("<div style='color:#60a5fa; font-size:0.75rem; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:14px;'>Label Distribution</div>", unsafe_allow_html=True)
    label_colors = {"Highly Recommended": "#4ade80", "Recommended": "#60a5fa", "Not Recommended": "#f87171"}
    label_counts = books_df["label"].value_counts()
    for lbl, cnt in label_counts.items():
        pct_val = round(cnt / total_books * 100)
        color = label_colors.get(lbl, "#94a3b8")
        st.markdown(f"""
        <div style='display:flex; justify-content:space-between; align-items:center;
                    background:#0a1628; border:1px solid #1a3060; border-radius:8px;
                    padding:10px 14px; margin-bottom:8px;'>
            <span style='font-size:0.8rem; color:#94a3b8;'>{lbl}</span>
            <span style='font-family:Syne,sans-serif; font-size:0.9rem; font-weight:700; color:{color};'>{cnt} <span style='font-size:0.7rem; color:#475569;'>({pct_val}%)</span></span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Model Section ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🤖 ML Models</div>', unsafe_allow_html=True)

if model_obj is None:
    st.warning("No trained model found. Open the sidebar and click **Train / Retrain Models**.")
else:
    # Fetch all models from DB
    _conn2 = get_db_connection()
    _cur2  = _conn2.cursor()
    _cur2.execute("SELECT model_name, created_at, feature_version, metrics_json, is_best FROM models ORDER BY created_at DESC")
    _rows  = _cur2.fetchall()
    _conn2.close()

    if _rows:
        # Pills row
        pills_html = '<div class="model-row">'
        for r in _rows:
            is_best = r["is_best"]
            cls = "model-pill best" if is_best else "model-pill"
            label = f'{r["model_name"]} {"★ Best" if is_best else ""}'
            pills_html += f'<div class="{cls}"><span class="dot"></span>{label}</div>'
        pills_html += '</div>'
        st.markdown(pills_html, unsafe_allow_html=True)

        # Active model metrics
        active_metrics_html = '<div class="panel"><div style="color:#60a5fa; font-size:0.75rem; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:16px;">Active Model Metrics</div><div class="metric-row">'
        for key, display in [("accuracy","Accuracy"),("precision","Precision"),("recall","Recall"),("f1_macro","F1 Macro"),("cv_f1_macro_mean","CV F1")]:
            val = model_metrics.get(key)
            if val is not None:
                active_metrics_html += f'<div class="metric-chip"><span class="val">{val:.3f}</span>{display}</div>'
        active_metrics_html += '</div></div>'
        st.markdown(active_metrics_html, unsafe_allow_html=True)

        # Full comparison table
        rows_data = []
        for r in _rows:
            m = json.loads(r["metrics_json"])
            rows_data.append({
                "Model":     ("★ " if r["is_best"] else "") + r["model_name"],
                "Accuracy":  round(m.get("accuracy",  0), 3),
                "Precision": round(m.get("precision", 0), 3),
                "Recall":    round(m.get("recall",    0), 3),
                "F1 Macro":  round(m.get("f1_macro",  0), 3),
                "CV F1":     round(m["cv_f1_macro_mean"], 3) if "cv_f1_macro_mean" in m else "-",
                "Trained":   r["created_at"],
            })
        st.dataframe(pd.DataFrame(rows_data), use_container_width=True, hide_index=True)

# ── Recommendation Form ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔍 Get Recommendations</div>', unsafe_allow_html=True)

available_courses = sorted(books_df["course"].unique().tolist())

st.markdown('<div class="form-panel">', unsafe_allow_html=True)
with st.form("recommend_form"):
    r1c1, r1c2, r1c3 = st.columns(3, gap="medium")
    with r1c1:
        course = st.selectbox("Course / Department", options=available_courses)
    with r1c2:
        semester = st.number_input("Semester", min_value=1, max_value=8, value=2, step=1)
    with r1c3:
        duration = st.selectbox("Study Duration", options=["Short-term", "Medium-term", "Long-term"])

    r2c1, r2c2, r2c3 = st.columns(3, gap="medium")
    with r2c1:
        top_n = st.slider("Results", 1, 10, 5)
    with r2c2:
        sort_by = st.selectbox("Sort By", options=["score","popularity","difficulty"],
            format_func=lambda x: {"score":"Score","popularity":"Popularity","difficulty":"Difficulty"}[x])
    with r2c3:
        only_highly = st.checkbox("Only Highly Recommended")

    submitted = st.form_submit_button("✦  Find Books")
st.markdown('</div>', unsafe_allow_html=True)

# ── Results ───────────────────────────────────────────────────────────────────
if submitted:
    if model_obj is None:
        st.error("No trained model. Open the sidebar and click **Train / Retrain Models**.")
    else:
        try:
            recs = recommend_books(
                course=course,
                semester=int(semester),
                study_duration=duration,
                top_n=top_n,
                only_highly_recommended=only_highly,
                sort_by=sort_by,
            )

            if not recs:
                st.markdown("""
                <div style='background:#0d1e38; border:1px solid #1e3a5f; border-radius:12px;
                            padding:24px; text-align:center; color:#64748b;'>
                    No matching books found. Try a different semester or uncheck the filter.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background:rgba(59,130,246,0.06); border:1px solid rgba(59,130,246,0.2);
                            border-radius:12px; padding:16px 20px; margin-bottom:20px;
                            display:flex; align-items:center; gap:12px;'>
                    <span style='font-size:1.4rem;'>✦</span>
                    <span style='color:#93c5fd; font-family:Syne,sans-serif; font-size:0.95rem; font-weight:600;'>
                        {len(recs)} recommendation{"s" if len(recs)>1 else ""} for
                        <span style='color:#60a5fa;'>{course}</span> ·
                        Semester <span style='color:#60a5fa;'>{semester}</span> ·
                        <span style='color:#60a5fa;'>{duration}</span>
                    </span>
                </div>
                """, unsafe_allow_html=True)

                for i, rec in enumerate(recs, start=1):
                    conf      = rec["confidence"]
                    conf_cls  = {"High":"conf-high","Medium":"conf-medium","Low":"conf-low"}.get(conf,"conf-low")
                    conf_dot  = {"High":"●","Medium":"●","Low":"●"}.get(conf,"●")
                    score_pct = int(rec["final_score"] * 100)
                    prob_pct  = int(rec["pred_prob_highly_recommended"] * 100)

                    tags = "".join([
                        f'<span class="tag">Sem {rec["semester"]}</span>',
                        f'<span class="tag">{rec["difficulty"]}</span>',
                        f'<span class="tag">{rec["duration_suitability"]}</span>',
                        f'<span class="tag">Score {rec["past_usage_score"]} ({rec["usage_bucket"]})</span>',
                        f'<span class="tag">{rec["label"]}</span>',
                    ])

                    st.markdown(f"""
                    <div class="rec-card">
                        <div class="rec-rank">#{i}</div>
                        <div class="rec-title">{rec["title"]}</div>
                        <div class="rec-course">{rec["course"]}</div>
                        <div class="rec-tags">{tags}</div>
                        <div style='display:flex; gap:10px; align-items:center; margin-bottom:12px;'>
                            <span class='confidence-badge {conf_cls}'>{conf_dot} {conf} confidence</span>
                            <span style='color:#475569; font-size:0.75rem;'>P(Highly Rec): {prob_pct}%</span>
                        </div>
                        <div class="rec-score-bar">
                            <div class="bar-label"><span>Final Score</span><span>{rec["final_score"]:.3f}</span></div>
                            <div class="bar-track"><div class="bar-fill" style="width:{score_pct}%;"></div></div>
                        </div>
                        <div class="rec-explanation">💡 {rec["explanation"]}</div>
                    </div>
                    """, unsafe_allow_html=True)

        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Recommendation failed: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<hr>
<div style='text-align:center; color:#2d3f5a; font-size:0.75rem; padding-bottom:20px;'>
    LibraryIQ &nbsp;·&nbsp; Feature version {FEATURE_VERSION} &nbsp;·&nbsp; Local ML demo — no external APIs
</div>
""", unsafe_allow_html=True)

conn.close()
