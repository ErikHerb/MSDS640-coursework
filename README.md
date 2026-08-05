# Data Science Capstone: Adult Census Income

Course: MSDS640 Data Science Capstone · Erik Herb

This repo holds my capstone project. The goal is to predict whether a person earns more than $50K a year from their demographic, education, and job information, using the UCI Adult Census Income data from the 1994 U.S. Census. I frame the use case as a lender or screening tool that flags likely high earners, so I care about which features drive the decision and how fair the model is across groups, not just raw accuracy.

The project is complete. All milestones are finished, and the final report, final notebook, and final presentation are in the repo.

## The three hypotheses, and how they turned out

The whole project is built around three hypotheses:

- **H1:** Education level is a stronger driver of the over-$50K prediction than hours worked. **Supported.**
- **H2:** Of the three models, XGBoost is the fairest across nativity, with the smallest US-born vs foreign-born recall gap on true high earners (smaller than logistic regression and the MLP). **Refuted.**
- **H3:** XGBoost classifies better than both the logistic regression baseline and the MLP on PR-AUC and F1. **Supported.**

H1 holds on all three trained models. Permutation importance on the held-out test set says education is worth roughly two and a half to three times as much as hours, and the 95 percent confidence interval on that difference excludes zero for every model.

H3 holds with four independent tests agreeing. XGBoost reaches 0.805 PR-AUC and 0.708 F1 on the test set against 0.774 and 0.684 for the tuned logistic regression. It won on all five cross-validation folds and in 1,000 of 1,000 bootstrap resamples, the paired t-test gives p = 0.0002, and McNemar's exact test gives p on the order of 1e-23. The part of H3 I do not claim is the MLP beating the logistic regression, because the bootstrap interval on that difference contains zero.

H2 is refuted. XGBoost has the largest nativity recall gap of the three at 0.079, not the smallest, and the same ordering shows up on sex. The more careful statement is that no model is statistically distinguishable from any other on this gap, because there are only 99 real foreign-born high earners in the test set and the confidence interval on a recall computed from 99 people is about 0.17 wide. Settling that comparison would take roughly 11 times as many foreign-born high earners. What is measurable is that all three models miss female high earners 11 to 15 points more often than male ones, and that gap is significant for every model.

The main finding is that the most accurate model was not the fairest, so the accuracy question and the fairness question had to be answered separately rather than one following from the other.

## What's in the data

The dataset is about 32,500 rows and 15 raw columns, one person per row. The target is income, either over $50K or $50K and under, and the classes are imbalanced at roughly 24 percent high earners. The raw file comes from the UCI Machine Learning Repository (the copy hosted on Kaggle as `adult.csv`). It is a single snapshot of the 1994 census year, so there is no time dimension and no temporal split applies. The data is public, released for research use under the UCI repository terms with no access restrictions, and anonymized at source.

The data is loaded into a small SQLite database (`adult.db`) that becomes the source of truth, and everything after that (cleaning, EDA, modeling) reads back out of it instead of the raw CSV. The `database/` folder has its own README with the exact build and pull-out steps.

After cleaning there are 32,537 rows and 13 columns, which becomes a 52 column model matrix once the eight categorical columns are one-hot encoded.

## Models and methods

Three models compete on the prediction task:

- **Logistic regression** as the interpretable baseline (built in the baseline notebook, and the null case H3 is measured against).
- **XGBoost** as the tree model.
- **MLP** (multilayer perceptron) as the neural network.

I also fit a **decision tree**, but not as a fourth competing model. It's an interpretability tool, suggested by my professor, that estimates feature importance with every feature competing for the same splits at once, which is what H1 needs and a one-feature-at-a-time view can't give.

Because the target is imbalanced, I lead with **PR-AUC and recall** rather than accuracy (a model that flags nobody as a high earner is still right about 76 percent of the time while catching zero of them). I handle the imbalance with **SMOTE oversampling on the training split only**, so no resampled rows ever leak into validation or test.

All three models are built inside the same imbalanced-learn pipeline, so any difference in the results comes from the model rather than from one of them getting better preprocessing. Each pipeline builds its own scaler rather than sharing one object, because a scikit-learn pipeline does not copy the steps it is handed, so sharing a scaler means refitting one model later silently changes the preprocessing of models already scored.

### Statistical methods

Every claim about one model being better than another is backed by a test with a confidence interval:

- **Paired t-test and Wilcoxon signed-rank** on five cross-validation fold scores. The fold indices are generated once and reused for all three models, which is what makes the scores paired.
- **McNemar's exact test** on the 6,508 shared test rows, which is the right test for two classifiers predicting on the same people.
- **Paired bootstrap confidence intervals**, 1,000 resamples, with every model scored on the identical resample so the difference between two models can be taken inside each replicate.

### Fairness work

The audit splits test performance by nativity and by sex and measures recall on real high earners inside each group, because that is the error that lands on the applicant rather than on the lender. Group accuracy is reported but not led with, since it is propped up by the base rate and points the opposite way. Three mitigation strategies are tested against the as-is model: training on the minority group only, group-balanced oversampling, and group-specific decision thresholds. Group-balanced oversampling is the one I recommend, because it recovers most of the gap without the deployed model ever branching on a protected attribute at prediction time.

## Repository structure

```
MSDS640-coursework/
├── data/
│   ├── raw/          adult.csv           (placeholder, see the setup steps below)
│   └── processed/    adult_pulled.csv    (placeholder for the table pulled from the database)
├── database/         adult.db + README   (the SQLite database, the source of truth)
├── notebooks/        the capstone notebooks, in milestone order
├── reports/          the written reports, the project plan, and the presentation decks
├── README.md         this file
└── requirements.txt  the packages needed to run the notebooks
```

I left out the `src/` and `results/` folders from the template, because all my code and plots live inside the notebooks rather than in separate `.py` scripts or exported output files.

## Project milestones and notebook order

The project follows the course's phase structure. All milestones are complete, and the notebooks below are listed in the order they are run. Each one builds on the last.

1. **Database Set Up and Initial Data Look** — Phase 1, Problem Definition and Project Plan (completed). Builds `adult.db` from the raw CSV, pulls the data back out, and does the first look at shape, types, missing values, and class balance. Paired in `reports/` with the project plan (`Data Science Capstone Project Plan.docx`), the initial data look and database setup report, and the first progress presentation.

2. **Project Data EDA and Feature Engineering** — Phase 2, Data Pre-processing, Feature Engineering, and EDA (completed). Missing-value handling, feature engineering, outlier analysis, the statistical tests behind the patterns, and the decision-tree feature importance for H1. Paired in `reports/` with the EDA and feature engineering report and the second progress presentation.

3. **Baseline Model** — Phase 3, Baseline Model (completed). Trains the logistic regression baseline with SMOTE, cross-validates it, tunes it, and evaluates it as the benchmark H3 is measured against. Paired with the baseline model report.

4. **Final Model Comparison and Fairness Audit** — Phase 4, Model Improvement, and Week 6, Final Report and Presentation (completed). Trains XGBoost and the MLP, tunes all three, runs the significance tests and bootstrap confidence intervals, does the nativity and sex fairness audit, tests the three mitigation strategies, and delivers a verdict on each hypothesis. The notebook is also written as the final report, with the markdown cells covering every required section end to end.

The final milestone ships as three files: the notebook, a written report PDF that stands alone without the code, and the final presentation deck.

## Where each required component lives

| Component | Location |
|---|---|
| Final report (written) | `reports/Data Science Capstone Final Report.pdf` |
| Final report (notebook form) | `notebooks/Data Science Capstone Final Model Comparison and Fairness Audit.ipynb` |
| Final presentation slides | `reports/Data Science Capstone Final Presentation.pptx` |
| Complete final notebook | same notebook as above, fully executed with all outputs |
| Environment file | `requirements.txt` |
| Repository overview | this file |

Inside the written report, the required sections map like this: title and abstract at the top; introduction and problem statement in section 1; research questions and hypotheses in section 2; the data in section 3; EDA in section 4; preprocessing in section 5; feature engineering and dimensionality reduction in section 6; the split strategy and the leakage check in section 7; methodology, models, metrics, and statistical methods in sections 8 through 11; results, tuning, cross-validation, and the statistical analysis in sections 12 through 18; the fairness audit and the mitigation strategies in sections 19 and 20; the three hypothesis verdicts in section 21; error analysis in section 22; overfitting and generalization in section 23; strengths, limitations, biases, societal impact, and takeaway lessons in section 24; the conclusion and future work in section 25; and reproducibility and references in section 26.

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
3. Put `adult.csv` in the same folder as the notebook you are running. Every notebook reads it with `pd.read_csv("adult.csv")` from its own working directory and rebuilds `adult.db` from it, so the file has to sit next to the notebook rather than in `data/raw/`. The copies under `data/` are empty placeholders, so download the file from the UCI repository or the Kaggle mirror linked below.
4. Open the notebooks in Jupyter and run them in the order listed above.

The packages are pandas, numpy, scikit-learn, imbalanced-learn (for SMOTE), xgboost, scipy, matplotlib, and seaborn. `sqlite3` ships with Python, so there's nothing to install for the database. The final notebook takes roughly 8 to 12 minutes to run end to end, most of which is the hyperparameter grid searches and the 5-fold cross-validation.

## Reproducibility

I set one fixed random seed (42) on the train/validation/test split, the cross-validation fold indices, SMOTE, and all three models, and I use a single load path from the database, so the notebooks rerun to the same numbers. The one manual step is putting `adult.csv` next to the notebook before running it, as described in step 3 above. After that each notebook runs top to bottom on a clean environment with no further steps, and the final notebook prints the library versions it ran on.

| Library | Version |
|---|---|
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| scikit-learn | 1.8.0 |
| imbalanced-learn | 0.14.2 |
| xgboost | 3.3.0 |
| scipy | 1.17.1 |

## Data source

Becker, B. and Kohavi, R. (1996). Adult. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/2/adult

Kaggle mirror of the same extract, which is where I pulled the CSV from: https://www.kaggle.com/datasets/uciml/adult-census-income

## Author

Erik Herb — M.S. Data Science, Bryant University. GitHub: [ErikHerb](https://github.com/ErikHerb)
