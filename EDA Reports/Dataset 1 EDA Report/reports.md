# EDA Executive Report: E-Commerce Product Return Prediction
**Dataset:** `Dataset 1` (`ecommerce_orders_dataset.csv`)  
**Scope:** Exploratory Data Analysis & Machine Learning Feasibility Assessment  
**Task:** Binary Classification — Predict Customer Return Propensity (`Returned` = Yes / No)

---

## 1. Dataset Health & Overview

| Metric | Observation | ML Implication |
| :--- | :--- | :--- |
| **Total Records** | **30,000 rows × 41 columns** | High statistical power; ample data for tree-based models and validation splits. |
| **Completeness** | **0% Missing Values** | Pristine data quality; no initial imputation required. |
| **Duplicates** | **0 Duplicate Rows** | Full record-level integrity across transactions. |
| **Customer Base** | **8,683 Unique Customers** | ~3.46 orders/customer; historical return rate can be engineered as a customer feature. |
| **Memory Footprint** | **9.4 MB (uncompressed)** | Fast in-memory training and low latency scoring. |

---

## 2. Target Variable Analysis (`Returned`)

* **Class Distribution:**
  * **Not Returned (`No`):** 26,967 orders (**89.89%**)
  * **Returned (`Yes`):** 3,033 orders (**10.11%**)
* **Class Imbalance Ratio:** **8.89 : 1** (~10% positive return rate).
* **Evaluation Alert:** **Do NOT use standard Accuracy** (a dummy model predicting all "No" achieves 89.9% accuracy with zero utility).
* **Recommended Metrics:** **PR-AUC (Average Precision)**, **ROC-AUC**, and **F1-Score / Recall at tuned decision thresholds**.

---

## 3. Key Behavioral & Segment Findings

1. **Product Category is the Strongest Domain Signal:**
   * **Fashion** exhibits the highest return rate at **13.87%** (845 / 6,092 orders), well above the 10.11% dataset average.
   * **Books** follow at **10.91%**, while **Home & Kitchen** (8.66%) and **Beauty** (8.79%) have the lowest return rates.
2. **Customer Segmentation Trends:**
   * **New Customers** are the most prone to return products (**11.53%**), compared to **Premium** (9.32%) and **Returning** (9.44%) customers.
3. **Order Financials & Distribution Outliers:**
   * `Order_Amount` (Mean: \$379.00, Median: \$171.03) and `Unit_Price` (Mean: \$126.82, Median: \$60.67) are heavily right-skewed with ~11–14% outliers; requires robust scaling or tree-based algorithms immune to monotonic monotonic transformations.
4. **Logistics & Delivery Duration:**
   * Average delivery duration is **4.44 days** (range: 0 to 13 days). There is a slight uptick in return rates for long delivery delays (>7 days), though overall linear correlation is weak ($r \approx 0.003$).

---

## 4. Critical Data Leakage & Feature Filtering

> [!CAUTION]
> **High-Risk Target Leakage Detected:**
> * **`Order_Status`**: Cross-tabulation confirms `Order_Status == 'Returned'` matches `Returned == 'Yes'` at **100% correlation**. Using `Order_Status` will cause 100% artificial accuracy and catastrophic production failure. **Must be dropped**.
> * **`Review_Rating`**: Correlates strongly with return ($r = -0.475$, mean rating 3.08 for returns vs 4.16 for kept items). However, reviews are submitted *after* delivery. If predicting return at checkout time, `Review_Rating` **must be excluded**.
> * **Post-Sale / Identifiers to Drop:** `Order_ID`, `Customer_ID`, `Product_ID`, `Order_Date`, `Profit_Amount`, and `High_Value_Order`.

---

## 5. Machine Learning Readiness Verdict

| Criterion | Status | Verdict |
| :--- | :---: | :--- |
| **Dataset Viability** | **READY** | High quality, 30k rows, zero nulls, clear commercial patterns. |
| **Recommended Pipeline** | **Scikit-learn / XGBoost** | One-Hot Encoding for categoricals + RobustScaler + LightGBM / XGBoost. |
| **Imbalance Strategy** | **Cost-Sensitive / Weighting** | Use `scale_pos_weight = 8.89` or SMOTE + Stratified 5-Fold Cross-Validation. |
| **Target Decision** | **Binary Target** | Map `Returned`: `{'No': 0, 'Yes': 1}`. |



## 6. Key Concerns Regarding Machine Learning Feasibility

1. **Weak Feature Discrimination:** Several numerical features, particularly `Delivery_Days` and `Discount_Percent`, show nearly identical distributions for returned and non-returned orders, suggesting limited standalone predictive power.

2. **Limited Variation Across Customer Segments:** Return rates across age groups and customer segments are relatively similar, potentially limiting their ability to distinguish high-risk orders.

3. **High Feature Redundancy:** Strong correlations exist between features such as `Order_Amount` and `Tax_Amount` (0.93), indicating potential redundancy and limited additional information from correlated variables.

4. **Class Imbalance:** Only 10.11% of orders are returned, resulting in an 8.89:1 class imbalance. This can make identifying returned orders challenging and requires appropriate evaluation metrics beyond accuracy.

5. **Uncertain Generalization:** Despite having 30,000 records, the dataset's ability to generalize to unseen real-world orders remains unproven. Leakage-free model evaluation, cross-validation, and learning curves are necessary to determine whether the available features provide sufficient predictive signal.

**Final Recommendation:** **APPROVED FOR EXPERIMENTAL ML MODELING; PRODUCTION READINESS PENDING VALIDATION.** The dataset contains 30,000 records with no missing values or duplicate rows, making it suitable for initial model development. However, observed EDA patterns do not guarantee strong predictive performance. Potential feature redundancy, class imbalance, and uncertain generalization must be evaluated through leakage-free training, cross-validation, and testing on representative unseen data before production deployment.
