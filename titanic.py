""" Import modules
:pandas- data analysis
:numpy- mathematical operations
:matplotlib.pyplot- data visualization on a chart
:seaborn- data visualization on a chart
:math- mathematical functions
:sys- interaction with the interpreter 
:os- file path operations
:sklearn- building predictive models
"""

import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import sys
import os 
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LinearRegression, LogisticRegression

model = LogisticRegression(C=0.1, max_iter=1000)


TRAIN_DATA_PATH = '/home/gosia/Pulpit/titanic_analysis_new/train.csv'
TEST_DATA_PATH = '/home/gosia/Pulpit/titanic_analysis_new/test.csv'
GENDER_SUBMISSION_PATH = '/home/gosia/Pulpit/titanic_analysis_new/gender_submission.csv'


def load_data (filepath) -> DataFrame:
    """
    Load data from csv file:
    :param filepath: (str) a filepath with input files
    :return: (DataFrame) input data
    """
    return pd.read_csv(filepath)


def preprocess_data (data) -> DataFrame:
    """
    Removes unnecessary columns, converts data types, fills in missing values
    :param data: (DataFrame) a Frame with data load
    :return: (DataFrame) preprocessing input data
    """

    columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    data = data.drop(columns_to_drop, axis=1)
    
    data['Age'] = data['Age'].astype(str).str.replace(',', '.').astype(float)
    data['Fare'] = data['Fare'].astype(float).round(2)
    data['Age'].fillna(data['Age'].median(), inplace = True)
    data['Fare'].fillna(data['Fare'].median(), inplace = True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace = True)
    data['Sex'] = data['Sex'].replace({'female': 1, 'male': 0})
    data['Embarked'] = data['Embarked'].replace({'S': 0, 'C': 1, 'Q': 2})
    
    return data

def plot_data (data):
    """
    Creates charts that compare various attributes with chances of survival.
    :param data: (DataFrame) a preprocessing input data
    """
    sns.set_style("whitegrid")
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    for feature in features:
        sns.barplot(x = feature, y = 'Survived', hue = feature, data = data, palette = 'Set2', legend = False)
        plt.title(f"Survival Rate by {feature}")  
        plt.ylabel("Survival Rate")  
        plt.xlabel(feature)
        plt.show()

def create_correlation_matrix (data):
    """
    Create a correlation matrix for given data
    :param data: (DataFrame) a preprocessing input data
    """
    corr_matrix = data.corr()
    print(corr_matrix['Survived'].sort_values(ascending = False))

def train_knn_model(X_train, Y_train, X_test):
    """
   It trains the KNN model and displays an error graph depending on the number of neighbors
    :param X_train: (DataFrame) trainig features set
    :param Y_train: (DataFrame) trening set labels
    :param X_test: (DataFrame) test features set
    :return: (ndarray) model of predictions
    """
    error_rates = []
    k_values = range(1, 10)
    for n in k_values:
        knn = KNeighborsClassifier(n_neighbors = n)
        knn.fit(X_train, Y_train)
        acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
        errorValue = 100 - acc_knn
        error_rates.append(errorValue)
    
    plt.plot(k_values, error_rates, marker = 'o', color = 'lightpink')
    plt.xlabel('Neighbors')
    plt.ylabel('Error Value')
    plt.title('Effect neighbors count on the value error')
    plt.show()
    
    best_knn = KNeighborsClassifier(n_neighbors = 5)
    best_knn.fit(X_train, Y_train)
    predictions = best_knn.predict(X_test)
    return predictions

def train_linear_regression_model(X_train, Y_train, X_test):
    """
    Trains a linear regression model and determines its accuracy
    :param X_train: (DataFrame) trainig features set
    :param Y_train: (DataFrame) trening set labels
    :param X_test: (DataFrame) test features set
    :return: (ndarray) model of predictions
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('featureselection', SelectKBest(k = 3)),
        ('Classifier', LinearRegression())
    ])
    pipeline.fit(X_train, Y_train)
    predictions = pipeline.predict(X_test).round(2)
    return predictions

def evaluate_model(predictions, real_values):
    """
    Evaluating the model using mean-square error.
    :param predictions: (ndarray) Model predictions
    :param real_values: (Series) actual values of the labels
    """
    mse = mean_squared_error(real_values, predictions)
    rmse = np.sqrt(mse)
    print("RMSE:", rmse)

def train_logistic_regression_model(X_train, Y_train, X_test, real_values):
    log_reg = LogisticRegression(max_iter=100)
    log_reg.fit(X_train, Y_train)
    predictions = log_reg.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(real_values, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(real_values, predictions))

def main():
   
    train_data = load_data(TRAIN_DATA_PATH)
    test_data = load_data(TEST_DATA_PATH)
    gender_submission = load_data(GENDER_SUBMISSION_PATH)
 
    processed_train_data = preprocess_data(train_data)
    processed_test_data = preprocess_data(test_data)
    
    plot_data(processed_train_data)
    create_correlation_matrix(processed_train_data)
    
    X_train = processed_train_data.drop('Survived', axis = 1)
    Y_train = processed_train_data['Survived']
    X_test = processed_test_data

    knn_predictions = train_knn_model(X_train, Y_train, X_test)
    lr_predictions = train_linear_regression_model(X_train, Y_train, X_test)
    evaluate_model(lr_predictions, gender_submission['Survived'])
    train_logistic_regression_model(X_train, Y_train, X_test, gender_submission['Survived'])

if __name__ == '__main__':
    main()
