"""
dashboard.py

A small Streamlit app that connects to main.py functions to:
 - show basic dataset statistics
 - show latest model metadata and metrics
 - allow user form input (course, semester, study duration) to receive top-N recommendations

Run locally:
  pip install -r requirements.txt    # requirements: pandas scikit-learn joblib streamlit
  python main.py --init-db           # creates library.db from books.json (only first time)
  python main.py --train             # train and save model (first time)
  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd

from main import (
    get_db_connection,
    load_books_from_db,
    load_latest_model,
    recommend_books,
    init_db,
)

st.set_page_config(page_title="Library Book Recommender", layout="centered")

st.title("Course & Duration-Based Library Book Recommender")
st.markdown(
    "Minimal local demo: choose course, semester and intended study duration to get recommended library books."
)

# Sidebar: allow DB initialization and model retrain triggers
st.sidebar.header("Admin / Maintenance")
if st.sidebar.button("Initialize DB from books.json (safe)"):
    init_db()
    st.sidebar.success("DB initialized (books.json loaded) — refresh main page.")

if st.sidebar.button("Reload latest model metadata"):
    # This button forces reloading model info below
    st.rerun()

# Load data
conn = get_db_connection()
books_df = load_books_from_db(conn)
model_obj, feature_cols, model_metrics = load_latest_model(conn)

st.header("Dataset summary")
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.write(f"Total books in DB: **{len(books_df)}**")
    st.write("Sample of books:")
    st.dataframe(books_df[["id", "title", "course", "semester", "difficulty", "duration_suitability", "past_usage_score", "label"]].head(10))
with col2:
    # count by course
    st.write("Books by course:")
    st.table(books_df["course"].value_counts().rename_axis("course").reset_index(name="count"))
with col3:
    st.write("Books by difficulty:")
    st.table(books_df["difficulty"].value_counts().rename_axis("difficulty").reset_index(name="count"))

st.header("Model information")
if model_obj is None:
    st.warning("No trained model found. Use the CLI: python main.py --train")
else:
    st.write(f"Model: **{model_obj.__class__.__name__}**")
    st.write("Feature columns used (snapshot):")
    st.write(", ".join(feature_cols))
    st.subheader("Metrics (stored with model)")
    st.json(model_metrics)

st.header("Get book recommendations")
with st.form("recommend_form"):
    # Provide choices derived from DB to avoid typos
    available_courses = sorted(books_df["course"].unique().tolist())
    course = st.selectbox("Course / Department", options=available_courses)
    semester = st.number_input("Semester (1..8)", min_value=1, max_value=8, value=2, step=1)
    duration = st.selectbox("Study duration", options=["Short-term", "Medium-term", "Long-term"])
    top_n = st.slider("How many recommendations?", 1, 10, 5)
    submitted = st.form_submit_button("Recommend")

if submitted:
    try:
        recs = recommend_books(course=course, semester=int(semester), study_duration=duration, top_n=top_n)
        if not recs:
            st.info("No matching books found for that course/semester range. Try different semester or broaden dataset.")
        else:
            st.success(f"Top {len(recs)} recommendations for {course}, semester {semester}, {duration}")
            # Display as table
            df_out = pd.DataFrame(recs)
            # Format columns for readability
            df_out_display = df_out[[
                "title", "semester", "difficulty", "duration_suitability", "past_usage_score",
                "pred_prob_highly_recommended", "final_score", "label"
            ]].rename(columns={
                "title": "Title",
                "semester": "Sem",
                "difficulty": "Difficulty",
                "duration_suitability": "Duration",
                "past_usage_score": "Popularity",
                "pred_prob_highly_recommended": "P(HighlyRec)",
                "final_score": "Score",
                "label": "CurrentLabel"
            })
            st.table(df_out_display.style.format({"P(HighlyRec)": "{:.2f}", "Score": "{:.3f}", "Popularity": "{:.0f}"}))
    except Exception as e:
        st.error(f"Recommendation failed: {e}")

# Footer / notes
st.markdown("---")
st.caption("Local demo — no external APIs. Model is simple and demonstrative (interpretability prioritized).")
conn.close()
