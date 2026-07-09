# Linear Regression & Regularization Project

## Project Overview

This project implements and compares various linear regression models from scratch and using `scikit-learn`. It explores the fundamental concepts of supervised learning, regularization techniques, and the challenge of overfitting. The primary objective is to build a robust predictive model for a regression task on a Kaggle dataset.

## Project Goals

*   To develop a deep, practical understanding of linear models for regression.
*   To implement core algorithms (Linear Regression, Ridge, Lasso, ElasticNet) from the ground up.
*   To learn how to evaluate models using MAE, RMSE, and R² metrics.
*   To apply data preprocessing, feature engineering, and normalization (MinMaxScaler, StandardScaler).
*   To diagnose and mitigate overfitting through regularization and normalization.
*   To compare the performance of custom implementations with established `sklearn` models.

## Technologies Used

*   **Language:** Python 3
*   **Libraries:**
    *   `pandas`, `numpy` for data manipulation and numerical operations.
    *   `matplotlib`, `seaborn` for data visualization.
    *   `scikit-learn` for model implementation, preprocessing, and evaluation.
    *   `collections` for data statistics.

## Project Structure

The project is organized as a Jupyter Notebook (`main.ipynb`) and follows a logical workflow:

1.  **Data Preparation:** Loading and cleaning the dataset, handling outliers.
2.  **Feature Engineering:** Parsing and creating binary features from a list of amenities.
3.  **Model Implementation from Scratch:** Building `LinearRegressionAnalytic`, `LinearRegressionGradient`, `LinearRegressionSGD`, `RidgeGD`, `LassoGD`, and `ElasticNetGD` classes.
4.  **Model Evaluation:** Calculating and comparing MAE, RMSE, and R² metrics for each model.
5.  **Normalization:** Implementing and applying MinMaxScaler and StandardScaler.
6.  **Overfitting Analysis:** Creating polynomial features to demonstrate overfitting and the effect of regularization.

## Results

The following table summarizes the final model performance on the test set after applying `StandardScaler` normalization. The **Lasso Regression** with `StandardScaler` emerged as the best and most stable model.

### Best Performance (Standard Scaler)

| Metric | Model | Test Score |
| :--- | :--- | :--- |
| **MAE** | Lasso_StandardScaler | 713.63 |
| **RMSE** | Lasso_StandardScaler | 1032.51 |
| **R²** | Lasso_StandardScaler | 0.57836 |

### Key Observations

*   `StandardScaler` significantly improved model performance for gradient-based methods.
*   `MinMaxScaler` was detrimental to all models.
*   Polynomial features introduced severe overfitting, which was partially mitigated by L1 and L2 regularization.
*   The custom implementations closely matched the performance of their `sklearn` counterparts, validating the understanding of the underlying algorithms.

## How to Run

1.  Ensure you have Python 3 installed along with the required libraries (pandas, numpy, matplotlib, seaborn, sklearn).
2.  Clone this repository.
3.  Download the dataset (train.json and test.json) and place it in the `data/` directory.
4.  Open the Jupyter Notebook (`main.ipynb`) and run all cells.

## What I Learned

This project was instrumental in solidifying my understanding of:

*   The mathematical formulations of ordinary least squares and regularized regression.
*   The practical differences between L1 and L2 regularization and their effects on feature selection.
*   The importance of feature scaling for gradient-based optimization algorithms.
*   The bias-variance tradeoff and how regularization can be used to find a better balance.
*   How to design and implement machine learning models from the ground up in Python.
