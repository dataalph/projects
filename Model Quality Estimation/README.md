# Model Quality Estimation: Validation, Feature Selection & Hyperparameter Tuning

## 1. Project Objective

The primary objective of this project is to develop a comprehensive, practical understanding of the machine learning model evaluation and optimization pipeline. The work focuses on three core areas:

1.  **Validation & Cross-Validation:** To implement and critically compare various data splitting schemes—including random, temporal (out-of-time), stratified, and grouped k-fold—to ensure robust model evaluation, mitigate overfitting, and prevent data leakage.
2.  **Feature Selection:** To explore and apply a diverse set of feature selection techniques—including Lasso (L1-regularization), correlation-based filtering, Permutation Importance, and SHAP (SHapley Additive exPlanations)—to enhance model performance and interpretability.
3.  **Hyperparameter Optimization:** To benchmark the efficiency and effectiveness of different hyperparameter tuning strategies, from exhaustive grid search and randomized search to sophisticated Bayesian optimization frameworks like Optuna.

This project serves as a hands-on demonstration of a complete, production-ready modeling pipeline, from raw data preprocessing to final model evaluation.

---

## 2. Data Source

The dataset for this project comes from a public Kaggle competition:

*   **Competition:** [Two Sigma Connect: Rental Listing Inquiries](https://www.kaggle.com/competitions/two-sigma-connect-rental-listing-inquiries/data)
*   **Provider:** Two Sigma
*   **Task:** Predict the popularity (`interest_level`) of a new rental listing on RentHop based on its content (text descriptions, photos, number of bedrooms, price, etc.).
*   **Data Origin:** The data is sourced from renthop.com, an apartment listing website focused on New York City.

The project uses the `train.json` and `test.json` files provided by the competition. The target variable, `interest_level`, has been converted to a continuous regression target (the `price` feature) for this analysis, allowing for a direct evaluation of numeric prediction accuracy.

---

## 3. Methodology & Tools

The project was executed using a structured, step-by-step methodology and a modern data science stack.

### 3.1. Data Preprocessing
- **Data Consolidation:** Training and test data were combined to allow for consistent feature engineering across the entire dataset.
- **Outlier Handling:** Price outliers were removed using percentile-based capping (1st and 99th percentiles). Anomalies in the `bathrooms` feature (values > 10) were also filtered out.
- **Feature Engineering:** A set of binary features (e.g., `Elevator`, `Doorman`, `LaundryinUnit`) was created from the provided `features` list, following the project specifications.

### 3.2. Validation Strategies
The project implemented and compared four custom cross-validation schemes against their `sklearn` equivalents:

*   **Random Split:** Used for baseline performance. Determines a model's ability to generalize to a random subset of data.
*   **Date (Out-of-Time) Split:** Respects the temporal order of data (using the `created` timestamp). This is crucial for time-series forecasting scenarios.
*   **K-Fold:** The standard k-fold cross-validation, providing a more stable performance estimate than a single random split.
*   **Stratified K-Fold:** Ensures that the distribution of the target variable (price, binned into quantiles) is preserved across all folds. This is essential for regression tasks with imbalanced target distributions.
*   **Group K-Fold:** Prevents data leakage by ensuring that samples from the same logical group (e.g., similar apartments based on `bedrooms`, `bathrooms`, and `amenities`) do not appear in both training and test sets across the same fold.
*   **Time Series Split:** Uses an expanding window approach, where the training set always precedes the test set temporally. This is the gold standard for validating forecasting models.

### 3.3. Feature Selection Methods
To identify the most predictive features and improve model efficiency, the following methods were applied:

*   **Lasso Coefficient Ranking (Embedded):** Leveraged the L1-regularization of a Lasso model to naturally shrink the coefficients of less important features to zero.
*   **Permutation Importance (Model-Agnostic):** Evaluated features by measuring the drop in model performance (R²) after randomly permuting their values. This method provides a direct measure of feature contribution.
*   **SHAP (Model-Agnostic):** Utilized SHAP values to provide a unified measure of feature importance based on game theory, explaining the contribution of each feature to individual predictions.

### 3.4. Hyperparameter Optimization
To find the optimal configuration for the ElasticNet model, three distinct optimization strategies were tested:

*   **Grid Search (Exhaustive):** Performed an exhaustive search over a predefined, coarse grid of hyperparameters (`alpha`, `l1_ratio`).
*   **Random Search:** Randomly sampled a wider range of hyperparameter values from continuous distributions for a fixed number of iterations.
*   **Bayesian Optimization (Optuna):** Employed the Tree-structured Parzen Estimator (TPE) sampler to intelligently explore the hyperparameter space, using prior evaluation history to guide future searches toward promising regions.

---

## 4. Results & Key Findings

### 4.1. Validation Strategy Comparison
A Linear Regression model was used as a baseline to compare different validation strategies. The primary metrics considered were test set R² and stability (the difference between train and test performance).

| Strategy | Test MAE | Test RMSE | Test R² | Difference (R²) | Key Observation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stratified K-Fold (sklearn)** | 712.59 | 1032.83 | **0.5796** | 0.0002 | **Best Stability & Top R²** |
| Random Split (sklearn) | 710.17 | 1017.06 | 0.5769 | 0.0036 | Best Test RMSE, but higher variance |
| Group K-Fold (sklearn) | 731.87 | 1055.80 | 0.5623 | 0.0177 | **Showed signs of overfitting** |
| Time Series Split (sklearn) | 713.03 | 1032.23 | 0.5785 | 0.0060 | Strong performance, respects temporal order |

**Takeaway:** **Stratified K-Fold** provided the optimal balance of high performance and low variance. It ensures that each fold is a representative sample of the entire dataset, making it the most reliable choice for this regression task. Group K-Fold clearly suffered from significant leakage due to the grouping strategy.

### 4.2. Feature Selection Method Comparison
Each method was evaluated after selecting the top 10 features and training a Lasso regression model.

| Method | Test RMSE | vs. Full Model | Speed | Stability | Key Observation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Full Lasso (Baseline)** | 1054.52 | 0% | Fast | High | All features included. |
| **Lasso (Weights)** | 1059.93 | +0.51% | Fast | High | **Best speed/quality trade-off.** |
| **Permutation Importance** | 1059.95 | +0.51% | Medium | Medium | Good selection, but computationally heavier. |
| **SHAP** | 1061.17 | +0.63% | **Slow** | **Highest** | Excellent for interpretation, but most expensive. |

**Takeaway:** Selecting the top 10 features using any method yielded a negligible performance drop (~0.5%). The Lasso-based feature ranking (`Lasso_top_10`) emerged as the best compromise between speed and accuracy, making it ideal for practical deployment. SHAP, while slower, provides unparalleled model explainability.

### 4.3. Hyperparameter Optimization Comparison
The ElasticNet model was tuned using different methods.

| Method | Test RMSE | Test R² | Speed | Key Observation |
| :--- | :--- | :--- | :--- | :--- |
| **ElasticNet_optuna** | **1054.48** | **0.5659** | Slow | **Best performing model.** |
| ElasticNet_optuna_cv | 1054.47 | 0.5659 | Slowest | Identical performance to Optuna, higher cost. |
| Grid Search | 1087.19 | 0.5386 | Slow | Poor due to coarse parameter grid. |
| Random Search | 1238.71 | 0.4010 | Fast | Very poor, indicating insufficient exploration. |

**Takeaway:** **Bayesian Optimization (Optuna) significantly outperformed both Grid Search and Random Search.** The Optuna model achieved a Test R² of **0.5659**, a substantial improvement over Grid Search (0.5386). This confirms that using an intelligent, adaptive search strategy is crucial for finding optimal hyperparameters in high-dimensional spaces. The addition of internal cross-validation (`optuna_cv`) did not provide a significant benefit while substantially increasing computation time.

---

## 5. Final Model & Conclusion

The final, optimized model is the **ElasticNet model tuned with Optuna** (without internal CV). This model strikes the best balance between performance, stability, and computational efficiency.

**Final Model Performance:**
*   **Test R²:** 0.5659
*   **Test MAE:** 720.44
*   **Test RMSE:** 1054.48

**Project Conclusions:**

1.  **Validation is Foundational:** Choosing the correct validation strategy is paramount. For this dataset, **Stratified K-Fold** proved to be the most robust, providing a stable and unbiased estimate of model performance. The use of Group K-Fold clearly demonstrated the dangers of unintended data leakage.
2.  **Feature Selection is a Powerful Tool:** High-quality features lead to better and more efficient models. Simple methods like **Lasso Coefficient Ranking** can effectively reduce feature space with minimal loss in performance, offering a significant speed/accuracy trade-off.
3.  **Bayesian Optimization is Superior:** Modern tuning frameworks like **Optuna** are far superior to traditional methods. They intelligently navigate the hyperparameter search space, finding better models (higher R²) with fewer evaluations compared to brute-force methods like Grid Search.

---

## 6. Technology Stack

*   **Language:** Python 3.10
*   **Core Libraries:** `pandas`, `numpy`
*   **Machine Learning:** `scikit-learn`
*   **Hyperparameter Tuning:** `optuna`
*   **Model Explainability:** `shap`
*   **Environment:** Jupyter Notebook