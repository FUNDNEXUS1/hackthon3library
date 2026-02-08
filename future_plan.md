🔮 Future Plans
===============

This document outlines **small, realistic, incremental improvements** to the current ETL and recommendation pipeline.\
The goal is to improve **reliability, clarity, and usefulness** without turning the project into something over-engineered.

Each task is designed to be achievable in **~15 minutes per day** with consistent, low-stress progress.

* * * * *

🧱 Current Pipeline (Baseline)
------------------------------

Right now, the system does:

-   **Extract**: Load book data from `books.json` into SQLite

-   **Transform**: Clean + encode features (course, semester, difficulty, duration, usage score)

-   **Load**: Store books and model metadata in `library.db`

-   **Train**: Train 3 models and pick the best one

-   **Serve**: Provide recommendations via Streamlit dashboard

This is a solid foundation. The improvements below build on this without breaking simplicity.

* * * * *

📅 Week 1 --- Data Quality & ETL Improvements
-------------------------------------------

**Goal:** Make the data pipeline more robust and transparent.

Small changes:

-   Add a **data validation step**:

    -   Check for missing fields (course, semester, label, etc.)

    -   Print warnings if rows are invalid

-   Add a **simple data summary log**:

    -   Number of books loaded

    -   Number of books per course

-   Add a **"last updated" timestamp** in the database for books

-   Add a CLI option:

    `python main.py --show-stats`

    to print dataset statistics in the terminal

Why this helps:

-   Makes the ETL step more trustworthy

-   Easier to explain in reports and demos

-   Helps catch bad data early

* * * * *

📅 Week 2 --- Better Feature Engineering (Still Simple)
-----------------------------------------------------

**Goal:** Slightly improve how data is transformed, without adding complexity.

Small changes:

-   Split `past_usage_score` into buckets:

    -   Low / Medium / High usage

-   Add a derived feature:

    -   `is_same_semester` (1 if book semester == input semester else 0)

-   Store **feature version** in model metadata (e.g., `"feature_version": "v2"`)

-   Log which features are used during training in a text file or DB

Why this helps:

-   Makes the "Transform" step more meaningful

-   Shows feature engineering in your ML pipeline

-   Still easy to maintain and explain

* * * * *

📅 Week 3 --- Training & Evaluation Improvements
----------------------------------------------

**Goal:** Make model training more transparent and comparable.

Small changes:

-   Save **all model results** (not just the best one) in the DB:

    -   Accuracy, Precision, Recall, F1

-   Add a simple **model comparison table** in the dashboard

-   Add **cross-validation (3-fold)** for one model (e.g., Random Forest)

-   Add a CLI option:

    `python main.py --list-models`

    to show all trained models and their metrics

Why this helps:

-   Shows a more realistic ML lifecycle

-   Makes evaluation more credible

-   Still keeps training time short and code simple

* * * * *

📅 Week 4 --- Recommendation & UX Improvements
--------------------------------------------

**Goal:** Improve how recommendations feel to the user.

Small changes:

-   Add **explanations** to recommendations:

    -   "Recommended because: high popularity + matches your semester"

-   Add a **confidence label**:

    -   High / Medium / Low confidence

-   Add a **filter toggle** in dashboard:

    -   "Only show Highly Recommended"

-   Add a **sort option**:

    -   By score

    -   By popularity

    -   By difficulty

Why this helps:

-   Makes the system feel more "intelligent" without changing the model

-   Improves user trust and clarity

-   Great for demos and presentations

* * * * *

🛠️ Small Technical Cleanup Tasks (Any Time)
--------------------------------------------

These can be done on any day when you want something easy:

-   Add more comments to tricky parts of the code

-   Rename variables for better readability

-   Add docstrings to functions that don't have them

-   Add a `requirements.txt`

-   Add basic error messages for:

    -   Empty database

    -   No trained model found

    -   Invalid user input

* * * * *

🎯 Long-Term (Optional, After One Month)
----------------------------------------

Only if you want to continue later:

-   Replace JSON seed with **CSV import support**

-   Add **simple feedback collection**:

    -   "Was this recommendation useful? Yes/No"

-   Store feedback and use it in retraining

-   Add a **very small data generator** to expand the dataset automatically

* * * * *

🧠 Guiding Principles
---------------------

-   Keep changes **small and reversible**

-   Prefer **clarity over cleverness**

-   Improve **one part of the pipeline at a time**

-   Always keep the project:

    -   Easy to run

    -   Easy to explain

    -   Easy to maintain

* * * * *

✅ Summary
---------

Over one month, with **~15 minutes a day**, this project can evolve from:

> A simple academic ML demo\
> to\
> A clean, well-structured, realistic ETL + ML recommendation system

...without ever becoming overcomplicated or fragile.
