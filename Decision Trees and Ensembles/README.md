# Supervised Learning. Decision Trees and Ensembles

## Project Overview

This project implements and compares various decision tree-based models for a binary classification task. It explores the fundamental concepts of tree-based learning, from a single CART (Classification and Regression Tree) to advanced ensemble methods including Random Forest and Gradient Boosting Decision Trees (GBDT). The project culminates in benchmarking three industry-standard GBDT implementations: LightGBM, CatBoost, and XGBoost.

The primary objective is to build a robust predictive model for the [Don't Get Kicked](https://www.kaggle.com/c/DontGetKicked) Kaggle competition, which involves predicting whether a purchased vehicle will be a "bad buy" for an auction company.

## Project Goals

* To develop a deep, practical understanding of decision tree algorithms and the CART (Classification and Regression Trees) methodology.
* To implement a custom Decision Tree Classifier and Regressor from scratch using NumPy, with Gini impurity and variance reduction as splitting criteria.
* To implement ensemble methods including Random Forest and Gradient Boosting Decision Trees (GBDT) using the custom tree as a base learner.
* To build an ExtraTreesClassifier with randomized threshold selection.
* To compare custom implementations against established `scikit-learn` models.
* To benchmark and analyze three modern GBDT implementations: LightGBM, CatBoost, and XGBoost.
* To identify key algorithmic differences between GBDT libraries and explain performance variations.

## Technologies Used

* **Language:** Python 3.11
* **Libraries:**
  * `pandas`, `numpy` for data manipulation and numerical operations.
  * `scikit-learn` for model implementation, preprocessing, and evaluation.
  * `lightgbm` for LightGBM implementation.
  * `catboost` for CatBoost implementation.
  * `xgboost` for XGBoost implementation.
  * `category_encoders` for advanced categorical feature encoding.
  * `matplotlib` (implied, for potential visualizations).

## Project Structure

The project is organized as a Jupyter Notebook (`main.ipynb`) and follows a logical workflow:

1. **Data Preparation:** Loading and cleaning the dataset, handling missing values (imputing zeros for numeric, "unknown" for categorical).
2. **Time-Based Split:** Creating a sequential train/validation/test split based on the `PurchDate` field to prevent data leakage and simulate a realistic time-series scenario.
3. **Preprocessing Pipeline:** Building a `ColumnTransformer` with OneHotEncoder for low-cardinality categoricals and CountEncoder with StandardScaler for high-cardinality categoricals.
4. **Custom Decision Tree Implementation:** Building a `Node` class and `DecisionTreeClassifier_custom` with support for `fit`, `predict`, and `predict_proba` methods.
5. **Custom Decision Tree Regressor:** Extending the classifier to support regression tasks using variance reduction (MSE) as the splitting criterion.
6. **Custom Random Forest:** Implementing bagging with bootstrap sampling and feature subsampling.
7. **Custom GBDT:** Implementing gradient boosting with logistic loss and incremental tree learning.
8. **Scikit-learn Benchmarking:** Comparing custom implementations against `sklearn`'s `DecisionTreeClassifier`.
9. **Modern GBDT Libraries:** Training and tuning LightGBM, CatBoost, and XGBoost models.
10. **Model Evaluation:** Comparing Gini scores across all models on the validation set.
11. **Final Model Testing:** Evaluating the best-performing model on the holdout test set.
12. **Bonus: ExtraTreesClassifier:** Implementing an extremely randomized tree classifier.

## Results

The following table summarizes the model performance on the validation dataset:

| Model | Validation Gini |
| :--- | :--- |
| DecisionTreeClassifier_custom | 0.4204 |
| DecisionTreeClassifier (sklearn) | 0.4206 |
| RandomForestClassifier_custom | 0.4699 |
| GBDTClassifier_custom | 0.4608 |
| LGBMClassifier | 0.4686 |
| XGBClassifier | 0.4742 |
| **CatBoostClassifier** | **0.4929** |

### Best Performance (CatBoostClassifier)

| Dataset | Gini Score |
| :--- | :--- |
| Train | 0.5867 |
| Validation | 0.4929 |
| Test | 0.4857 |

### Key Observations

* **CatBoost achieved the highest validation Gini score (0.4929)**, outperforming LightGBM and XGBoost by a notable margin.
* **CatBoost's victory is attributed to:**
  * **Ordered Target Encoding:** Handles categorical features without target leakage, particularly important given the class imbalance.
  * **Symmetric Trees:** Acts as a built-in regularizer, preventing overfitting to outliers in the rare positive class.
  * **Ordered Boosting:** Provides unbiased gradient estimates, stabilizing training on imbalanced data.
* **Custom implementations performed well** but were limited by their pure Python implementation and lack of sophisticated categorical handling.
* **XGBoost and LightGBM** performed similarly, with XGBoost slightly edging out LightGBM on this dataset.
* **The model shows moderate overfitting** with a drop from train Gini (0.5867) to test Gini (0.4857), but the consistent performance across validation and test suggests good generalization.

### Why CatBoost Won

CatBoost's architecture is inherently better suited for datasets with:
1. **High-cardinality categorical features** (`Make`, `Model`, `Trim`, `SubModel`, `VNST`)
2. **Severe class imbalance** (approx. 11% positive class)
3. **Time-based data** where target leakage in encoding would be catastrophic

Unlike LightGBM, whose primitive target encoding generates noise under class imbalance, CatBoost delivers a statistically honest signal through Ordered Target Encoding. XGBoost, on the other hand, cannot handle categorical features natively, requiring manual encoding that is almost always inferior to CatBoost's built-in solution.

## How to Run

1. Ensure you have Python 3.11 installed along with the required libraries:
   ```
   pip install pandas numpy scikit-learn lightgbm catboost xgboost category_encoders
   ```

2. **Download the dataset:** The data for this project comes from the [Don't Get Kicked](https://www.kaggle.com/c/DontGetKicked) competition on Kaggle. You will need to download the `train.csv` file from the competition's data page.

3. Place the downloaded `train.csv` file in the `../datasets/` directory relative to the notebook (or adjust the path in the notebook).

4. Open the Jupyter Notebook (`main.ipynb`) and run all cells.

5. The `my_utils.py` module must be in the same directory as the notebook, containing the `time_ordered_split_3_new` and `update_model_results` functions.

## What I Learned

This project was instrumental in solidifying my understanding of:

* The mathematical foundations of decision trees, including Gini impurity, information gain, and variance reduction.
* The mechanics of the CART algorithm and its recursive partitioning approach.
* How bagging (Random Forest) and boosting (GBDT) improve upon single tree models through different ensemble strategies.
* The practical differences between GBDT implementations and their unique algorithmic innovations.
* The importance of handling categorical features properly, particularly in the presence of class imbalance.
* The bias-variance tradeoff and how ensemble methods, regularization, and tree growth strategies control overfitting.
* How to design and implement machine learning models from the ground up in Python and benchmark them against industry-standard libraries.
* The tradeoffs between interpretability (single decision tree) and predictive performance (ensemble methods).