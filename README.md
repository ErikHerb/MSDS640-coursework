# Data Science Capstone: Adult Census Income

Course: MSDS640 Data Science Capstone · Erik Herb

This repo holds my capstone project. The goal is to predict whether a person earns more than $50K a year from their demographic, education, and job information, using the UCI Adult Census Income data from the 1994 U.S. Census. I frame the use case as a lender or screening tool that flags likely high earners, so I care about which features drive the decision and how fair the model is across groups, not just raw accuracy.

## The three hypotheses

The whole project is built around three hypotheses:

- **H1:** Education level is a stronger driver of the over-$50K prediction than hours worked.
- **H2:** Of the three models, XGBoost is the fairest across nativity, with the smallest US-born vs foreign-born recall gap on true high earners (smaller than logistic regression and the MLP).
- **H3:** XGBoost classifies better than both the logistic regression baseline and the MLP on PR-AUC and F1.

## What's in the data

The dataset is about 32,500 rows and 15 raw columns, one person per row. The target is income, either over $50K or $50K and under, and the classes are imbalanced at roughly 24 percent high earners. The raw file comes from the UCI Machine Learning Repository (the copy hosted on Kaggle as `adult.csv`).

The data is loaded into a small SQLite database (`adult.db`) that becomes the source of truth, and everything after that (cleaning, EDA, modeling) reads back out of it instead of the raw CSV. The `database/` folder has its own README with the exact build and pull-out steps.

## Models and methods

Three models compete on the prediction task:

- **Logistic regression** as the interpretable baseline (built in the baseline notebook, and the null case H3 is measured against).
- **XGBoost** as the tree model.
- **MLP** (multilayer perceptron) as the neural network.

I also fit a **decision tree**, but not as a fourth competing model. It's an interpretability tool, suggested by my professor, that estimates feature importance with every feature competing for the same splits at once, which is what H1 needs and a one-feature-at-a-time view can't give.

Because the target is imbalanced, I lead with **PR-AUC and recall** rather than accuracy (a model that flags nobody as a high earner is still right about 76 percent of the time while catching zero of them). I handle the imbalance with **SMOTE oversampling on the training split only**, so no resampled rows ever leak into validation or test.

## Repository structure

```
MSDS640-coursework/
├── data/
│   ├── raw/          adult.csv           (the original file, untouched)
│   └── processed/    adult_pulled.csv    (the cleaned table pulled from the database)
├── database/         adult.db + README   (the SQLite database, the source of truth)
├── notebooks/        the capstone notebooks, in milestone order
├── reports/          the written reports, the project plan, and the progress decks
├── README.md         this file
└── requirements.txt  the packages needed to run the notebooks
```

I left out the `src/` and `results/` folders from the template for now, because all my code and plots live inside the notebooks rather than in separate `.py` scripts or exported output files. I'll add them if a later milestone pulls reusable functions out into scripts.

## Notebooks, in order

Run them top to bottom in this order, since each one builds on the last:

1. **Database Set Up and Initial Data Look** — builds `adult.db` from the raw CSV, pulls the data back out, and does the first look at shape, types, missing values, and class balance.
2. **Project Data EDA and Feature Engineering** — the full EDA: missing-value handling, feature engineering, outlier analysis, the statistical tests behind the patterns, and the decision-tree feature importance for H1.
3. **Baseline Model** — trains the logistic regression baseline with SMOTE, cross-validates it, tunes it, and evaluates it as the benchmark H3 is measured against.

## Reports and presentations

The `reports/` folder has the written write-ups (the initial data look and database setup report, and the EDA and feature engineering report), the project plan, and the two progress presentations.

## How to set it up and run

1. Clone the repo:
   ```bash
   git clone https://github.com/ErikHerb/MSDS640-coursework.git
   cd MSDS640-coursework
   ```
2. Install the packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebooks in Jupyter and run them in the order listed above. The first notebook rebuilds `adult.db` from `adult.csv`, and the rest read from the database.

The main packages are pandas, numpy, scikit-learn, imbalanced-learn (for SMOTE), matplotlib, and seaborn. XGBoost comes in a later milestone when H2 and H3 are tested. `sqlite3` ships with Python, so there's nothing to install for the database.

## Reproducibility

I set one fixed random seed and use a single load path from the database, so the notebooks rerun to the same numbers. The baseline notebook prints the library versions it ran on.

## Author

Erik Herb — M.S. Data Science, Bryant University. GitHub: [ErikHerb](https://github.com/ErikHerb)
