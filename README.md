# Introduction

This project is focused on analyzing and predicting the survival rates of passengers aboard the Titanic. Leveraging a combination of Python libraries for data manipulation, analysis, and predictive modeling, we aim to uncover insights into the factors that influenced survival and predict outcomes for a set of passengers.

## Libraries Used

- **Pandas**: efficient data analysis and manipulation.
- **NumPy**: performing mathematical operations.
- **Matplotlib.pyplot & Seaborn**: data visualization through various charts, enhancing the understanding of data relationships.
- **Math**: utilize mathematical functions.
- **Sys & OS**: interaction with the Python interpreter and operating system, allowing for file path operations.
- **Scikit-learn (sklearn)**: building predictive models, including preprocessing, model selection, and evaluation.

## Dataset

The project utilizes a dataset split into three parts:
- Training data (`train.csv`): Used to train the models.
- Test data (`test.csv`): Used to predict survival outcomes.
- Gender submission (`gender_submission.csv`): Provides a template for submission to a competition or for evaluating model predictions.

## Features

- Data Loading and Preprocessing: Cleansing the dataset by removing unnecessary columns, converting data types, and handling missing values.
- Data Visualization: Creating charts to compare various attributes against the chances of survival.
- Correlation Analysis: Identifying the strength of relationships between variables.
- Predictive Modeling: Developing models using K-Nearest Neighbors (KNN), Linear Regression, and Logistic Regression to predict survival rates.
- Model Evaluation: Assessing model performance through metrics like mean square error, classification reports, and confusion matrices.

## Predictive Models

1. **K-Nearest Neighbors (KNN)**: Evaluates the impact of neighbor count on prediction accuracy.
2. **Linear Regression**: Predicts survival based on linear relationships between variables.
3. **Logistic Regression**: Utilized for binary classification to predict the likelihood of a passenger's survival.

## Usage

The main script orchestrates the data loading, preprocessing, visualization, model training, and evaluation processes. Run the script using a Python interpreter to execute the project pipeline:

```bash
python titanic_survival_prediction.py

