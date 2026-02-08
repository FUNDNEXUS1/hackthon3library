📚 Course & Duration-Based Library Book Recommendation System
=============================================================

A simple, local, and academic-friendly machine learning project that recommends university library books to students based on:

-   Course / Department

-   Semester

-   Intended study duration (Short / Medium / Long-term)

The goal of this project is **not** to build a commercial-grade recommender, but to demonstrate a **clear, reproducible, end-to-end ML pipeline** suitable for coursework, hackathons, and mid-term evaluations.

* * * * *

✨ Features
----------

-   📦 Local dataset stored in SQLite (`library.db`)

-   🧪 Reproducible seed data from `books.json`

-   🤖 Trains and compares:

    -   Logistic Regression

    -   Decision Tree

    -   Random Forest

-   🏆 Automatically selects and saves the best model

-   📊 Stores model metadata (version, metrics, features) in DB

-   🖥️ Simple Streamlit dashboard for:

    -   Viewing dataset summary

    -   Viewing model metrics

    -   Getting book recommendations interactively

-   🔁 Supports retraining when data changes

* * * * *

🗂️ Project Structure
---------------------

```bash
├── books.json        # Seed dataset (reproducible)
├── main.py           # Core logic: DB, training, evaluation, recommendation
├── dashboard.py      # Streamlit UI
├── library.db        # SQLite database (auto-created)
└── models/           # Saved trained models (auto-created)`
```

* * * * *

⚙️ Requirements
---------------

-   Python 3.8+

-   Packages:

    -   pandas

    -   scikit-learn

    -   joblib

    -   streamlit

Install them with:

`pip install pandas scikit-learn joblib streamlit`

* * * * *

🚀 How to Run (Step by Step)
----------------------------

> Make sure you are in the project folder and your virtual environment is activated (if you use one).

### 1️⃣ Initialize the Database

This creates `library.db` and loads data from `books.json`:

`python main.py --init-db`

You should see:

`Init DB: inserted XX new books from books.json`

* * * * *

### 2️⃣ Train the Machine Learning Model

This will:

-   Load data from the database

-   Train 3 models

-   Evaluate them

-   Select the best one

-   Save it to the `models/` folder

-   Store model metadata in the database

Run:

`python main.py --train`

You should see output like:

`[train] LogisticRegression --- f1_macro: ...
[train] DecisionTree --- f1_macro: ...
[train] RandomForest --- f1_macro: ...
[train] Best model: RandomForest saved to models/RandomForest_....joblib`

* * * * *

### 3️⃣ (Optional) Quick Console Test

This runs an end-to-end test in the terminal:

`python main.py --quick-test`

It will print a few recommended book titles.

* * * * *

### 4️⃣ Start the Dashboard (UI)

Run:

`streamlit run dashboard.py`

Your browser will open at:

`http://localhost:8501`

* * * * *

🖥️ Using the Dashboard
-----------------------

On the web page you can:

-   View dataset statistics

-   See which model is currently active and its metrics

-   Enter:

    -   Course

    -   Semester

    -   Study duration

-   Click **Recommend** to get top-N book suggestions

The results show:

-   Book title

-   Difficulty

-   Duration type

-   Popularity score

-   Model probability

-   Final ranking score

* * * * *

🔁 Updating Data and Retraining
-------------------------------

If you:

-   Edit `books.json`, or

-   Add new books to the database

Then simply run:

`python main.py --init-db
python main.py --train
streamlit run dashboard.py`

This will:

-   Update the DB

-   Retrain the model

-   Use the new model in the dashboard

* * * * *

🧠 How the Recommendation Works (Simple Explanation)
----------------------------------------------------

1.  The model learns patterns from:

    -   Course

    -   Semester

    -   Difficulty

    -   Study duration

    -   Past usage score

2.  For each book, it predicts how likely it is to be **"Highly Recommended"**

3.  The final ranking score combines:

    -   Model prediction

    -   Book popularity (past usage score)

4.  Books are sorted by this final score and shown to the user

* * * * *

🎯 Project Goals
----------------

-   Demonstrate a complete ML lifecycle:

    -   Data → Training → Evaluation → Saving → Prediction → UI

-   Keep everything:

    -   Local

    -   Reproducible

    -   Easy to understand

    -   Easy to maintain

-   Suitable for:

    -   Academic projects

    -   Hackathons

    -   ML pipeline demonstrations

* * * * *

📜 License
----------

MIT Lincese.
This project is for educational and academic use.\
You are free to modify and extend it for learning purposes.
