# Credit Card Fraud Detection Using Machine Learning

## Project Overview

This project explores credit card transaction data and builds a machine learning classification model to identify potentially fraudulent transactions. It uses Python for data analysis and Scikit-learn to train and evaluate a Logistic Regression model.

The project covers data exploration, data quality checks, duplicate removal, feature and target separation, train-test splitting, feature scaling, class-imbalance handling through random undersampling, model training, and classification performance evaluation.

The primary objective is to identify fraudulent transactions while examining the trade-off between detecting fraud and incorrectly flagging legitimate transactions.

## Objectives

- Explore the structure and characteristics of credit card transaction data.
- Inspect dataset dimensions, column names, data types, and summary statistics.
- Identify missing values and duplicate records.
- Examine the distribution of the target variable to understand class imbalance.
- Perform univariate and multivariate exploratory data analysis.
- Separate input features from the target variable.
- Split the dataset into training and testing subsets.
- Standardize features using `StandardScaler`.
- Apply random undersampling to balance the training classes.
- Train a Logistic Regression classifier.
- Evaluate model performance using a confusion matrix, accuracy, precision, recall, F1-score, classification report, and ROC-AUC.

## Technologies and Libraries

- **Python** — programming language
- **Pandas** — data loading, manipulation, inspection, and statistical summaries
- **Scikit-learn** — data splitting, feature scaling, Logistic Regression, and model evaluation
- **imbalanced-learn** — random undersampling to address class imbalance
- **Matplotlib** — visualization through the Pandas histogram plotting interface
- **Google Colab / Jupyter Notebook** — interactive development and analysis
- **CSV** — dataset storage format

## Dataset

The notebook loads the dataset from `creditcard.csv`.

The initial dataset inspection reports **25,838 rows and 31 columns**.

### Dataset columns

- `Time` — transaction time feature.
- `V1` to `V28` — numerical transaction features supplied in the dataset.
- `Amount` — transaction amount.
- `Class` — target variable used for classification.

The target variable uses two class labels:

- `0` — non-fraudulent transaction.
- `1` — fraudulent transaction.

The dataset contains significantly more non-fraudulent transactions than fraudulent transactions, making class imbalance an important consideration during model training.

**Dataset note:** The CSV file is not included in this repository by default. Place the dataset in the project directory before running the notebook. Check the dataset's original source and applicable usage terms before redistributing it.

## Project Workflow

### 1. Data Loading and Inspection

Pandas is used to load the CSV file into a DataFrame.

The notebook uses the following operations:

- `pd.read_csv()` to load the dataset.
- `df.head()` to inspect the first five records.
- `df.shape` to check the number of rows and columns.
- `df.columns` to inspect the column names.
- `df.info()` to review data types, non-null counts, and memory usage.

These steps provide an initial understanding of the dataset's structure and quality.

### 2. Data Quality Assessment

The notebook checks for missing values and duplicate records using:

- `df.isnull()`
- `df.isnull().sum()`
- `df.duplicated()`
- `df.duplicated().sum()`

The initial duplicate check reports **100 duplicate records**. The notebook then uses `df.drop_duplicates(inplace=True)` to remove duplicate rows and checks the duplicate count again.

The notebook also checks missing values in the `Class` column and includes a step to remove records with a missing target value.

### 3. Target Variable Analysis

The project examines the target distribution using:

- `df["Class"].value_counts()`
- `df["Class"].unique()`

This analysis helps identify the imbalance between legitimate and fraudulent transactions and confirms the target labels present in the data.

### 4. Exploratory Data Analysis (EDA)

The notebook contains both univariate and multivariate analysis.

**Univariate analysis**

The project uses `df["Amount"].describe()` to calculate descriptive statistics, including:

- Count
- Mean
- Standard deviation
- Minimum and maximum
- 25th, 50th, and 75th percentiles

It also plots a histogram of transaction amounts using `df["Amount"].hist()` to inspect their distribution.

The recorded summary reports a minimum transaction amount of `0.00`, a median of `18.98`, a mean of approximately `76.00`, and a maximum of `7,879.42` for the data used in that output.

**Multivariate analysis**

The notebook calculates a numerical correlation matrix using `df.corr(numeric_only=True)`. This allows relationships between numerical variables to be inspected.

### 5. Feature and Target Separation

The notebook separates the predictor variables from the target:

- `X` contains the input features, excluding `Class`.
- `Y` contains the `Class` target variable.

This separation prepares the data for supervised classification.

### 6. Training and Testing Split

The project uses Scikit-learn's `train_test_split()` with:

- `random_state=42` for reproducibility.
- `stratify=Y` to preserve the target class proportions across the split.

The recorded evaluation output contains **6,435 test records**, including 6,413 non-fraudulent and 22 fraudulent examples.

### 7. Feature Scaling

The notebook uses Scikit-learn's `StandardScaler` to standardize numerical input features before training the classifier.

Scaling is important for Logistic Regression because features with different numerical ranges can affect model optimization.

**Implementation note:** The notebook currently calls `fit_transform()` separately on both the training and testing data. The recommended approach is to fit the scaler on the training data and use `transform()` on the test data to maintain a consistent transformation and avoid test-data leakage.

### 8. Class Imbalance Handling

The notebook uses `RandomUnderSampler` from `imbalanced-learn`.

Random undersampling reduces the number of majority-class examples in the training data so that the model trains on a more balanced set.

The recorded output shows **66 examples in each class** after undersampling.

This approach helps the classifier pay more attention to the minority fraud class. However, undersampling discards some majority-class observations, and its effect on model performance should be evaluated carefully.

### 9. Model Training

The project trains a Logistic Regression classifier using Scikit-learn.

The model is initialized with `max_iter=1000` and fitted to the undersampled training data.

Logistic Regression is used as a baseline classification algorithm for distinguishing between legitimate and potentially fraudulent transactions.

### 10. Model Evaluation

The notebook evaluates predictions using several metrics:

- **Confusion matrix** — summarizes correct and incorrect classifications.
- **Accuracy** — proportion of all predictions that are correct.
- **Precision** — proportion of transactions predicted as fraudulent that are actually fraudulent.
- **Recall** — proportion of actual fraudulent transactions correctly identified.
- **F1-score** — harmonic mean of precision and recall.
- **Classification report** — summarizes precision, recall, F1-score, and support for each class.
- **ROC-AUC** — evaluates how well the predicted probabilities distinguish between the two classes across classification thresholds.

These metrics are especially important for fraud detection because accuracy alone can be misleading when the target classes are highly imbalanced.

## Recorded Model Results

The notebook contains the following evaluation output:

| Metric | Recorded result |
|---|---:|
| Accuracy | 97.93% |
| Precision | 12.75% |
| Recall | 86.36% |
| F1-score | 22.22% |
| ROC-AUC | 0.9935 |

The recorded confusion matrix is:

| | Predicted non-fraud | Predicted fraud |
|---|---:|---:|
| Actual non-fraud | 6,283 | 130 |
| Actual fraud | 3 | 19 |

### Interpretation

- The recorded recall of **86.36%** indicates that the model identified 19 of the 22 fraudulent transactions in the test set.
- The recorded precision of **12.75%** indicates that many transactions flagged as fraudulent were actually non-fraudulent.
- The recorded accuracy of **97.93%** should be interpreted alongside precision and recall because the dataset is imbalanced.
- The recorded ROC-AUC is high, but it should be re-evaluated after the preprocessing and scaling workflow is corrected.

**Important:** These figures are the results saved in the notebook, not a claim of independently verified or production-ready model performance. The notebook contains inconsistencies in the order of data cleaning and feature creation, as well as test-set scaling. Rerun the corrected pipeline before reporting these figures as final results.

## How to Run the Project

### Prerequisites

Install Python and the required libraries.

```bash
pip install pandas scikit-learn imbalanced-learn matplotlib jupyter
```

### Setup

1. Clone or download this repository.
2. Place `creditcard.csv` in the project directory.
3. Open the notebook in Jupyter Notebook or Google Colab.
4. Update the CSV file path to match your environment.
5. Run the notebook cells in order, after correcting the preprocessing workflow described below.

For a local setup, the dataset can be loaded with:

```python
import pandas as pd

df = pd.read_csv("creditcard.csv")
```

## Known Limitations and Recommended Improvements

- **Data-cleaning order:** Remove or otherwise handle rows with missing target values and remove duplicates before creating `X` and `Y`.
- **Consistent scaling:** Fit `StandardScaler` on the training data only, then transform the test data with the same fitted scaler.
- **Reproducible evaluation:** Rerun every downstream cell after changing the cleaning or preprocessing steps so the saved outputs correspond to the final pipeline.
- **Class imbalance:** Compare random undersampling with other approaches and evaluate performance on an appropriately representative test set.
- **Model comparison:** Evaluate additional classifiers, such as Decision Tree or Random Forest, as potential alternatives to Logistic Regression.
- **Threshold selection:** Examine the precision-recall trade-off before selecting a fraud-detection threshold.
- **Further analysis:** Investigate feature distributions and the relationship between transaction features and the target class.

## Key Learnings

This project provides practical experience with:

- Loading and inspecting structured data using Pandas.
- Checking missing values and duplicate records.
- Performing descriptive statistical analysis and correlation analysis.
- Visualizing numerical data distributions.
- Preparing features and target labels for supervised learning.
- Splitting data into training and testing sets.
- Standardizing numerical features.
- Handling class imbalance using random undersampling.
- Training a Logistic Regression classifier.
- Evaluating imbalanced classification with precision, recall, F1-score, confusion matrix, and ROC-AUC.
- Interpreting the limitations of model evaluation results.

## Project Status

**Implemented in the notebook:** Initial data exploration, data quality checks, duplicate removal, descriptive analysis, correlation analysis, feature preparation, train-test splitting, feature scaling, random undersampling, Logistic Regression training, and model evaluation.

**Recommended next step:** Correct the preprocessing order and scaling workflow, rerun the notebook, and confirm the updated metrics before treating the model results as final.

## Conclusion

This project demonstrates a machine learning workflow for exploring credit card transaction data and building a baseline fraud-classification model. It highlights the importance of data quality, class-imbalance handling, feature preprocessing, and using multiple evaluation metrics when assessing fraud-detection performance.

The project can be extended through improved preprocessing, model comparisons, and more detailed analysis of false positives and missed fraudulent transactions.
