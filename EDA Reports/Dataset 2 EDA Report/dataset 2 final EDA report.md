# EDA Executive Report: Indian E-Commerce Return Prediction
**Dataset:** `Dataset 2` (Multi-Table Relational E-Commerce Intelligence Dataset)  
**Scope:** Relational Extraction, Exploratory Data Analysis & Machine Learning Feasibility Assessment  
**Task:** Binary Classification — Predict Customer Return Propensity (`Returned` = Yes / No, `is_returned` = 1 / 0)

---

## 1. Relational Extraction & Dataset Health

The dataset was extracted by joining all 9 relational tables centered on `returns.csv` and consolidating orders into a master classification dataset:

| Dimension | Observation | Engineering & ML Implication |
| :--- | :--- | :--- |
| **Relational Scale** | **9 Tables Joined** (25,000 customers, 550 products, 100,000 orders, 254,331 items) | Complete coverage of customer lifecycle, logistics, payments, and product catalog. |
| **Referential Integrity** | **100% Integrity (0 Orphaned Records)** | Foreign keys join perfectly across orders, customers, and returns. |
| **Classification Scale** | **100,000 Orders × 51 Features** | High statistical depth; 3.3x larger than Dataset 1. |
| **Customer Universe** | **22,820 Unique Active Customers** | ~4.38 orders/customer; enables powerful customer RFM and historical return rate features. |
| **Missing Values** | **Contextual & Structural Missingness** | Non-campaign orders (82.8% null in campaign cols); unfulfilled orders (10.3% null in delivery); demographic nulls are under 1% MCAR. |

---

## 2. Target Variable Analysis (`Returned` / `is_returned`)

The target variable is derived by linking transaction IDs to `returns.csv`:

* **Class Breakdown:**
  * **Not Returned (`No` / `0`):** 90,490 orders (**90.49%**)
  * **Returned (`Yes` / `1`):** 9,510 orders (**9.51%**)
* **Class Imbalance Ratio:** **9.52 : 1** (~1:10 positive return prevalence, highly consistent with Dataset 1's 10.11%).
* **Metric Directive:** Standard **Accuracy is invalid** (a trivial baseline achieves 90.5% with 0% return recall).
* **Mandatory Evaluation Suite:** **PR-AUC (Precision-Recall AUC)**, **ROC-AUC**, and **F1-Score / Recall at optimized probability decision thresholds**.

---

## 3. Deep-Dive Return Analytics (From `returns.csv`)

Analyzing the **12,075 returned line items** reveals the operational and commercial drivers of customer returns:

### A. Root Cause Breakdown
1. **Size Issue:** **18.23%** (2,201 returns) — Sizing ambiguity remains the single largest return trigger.
2. **Changed Mind:** **16.43%** (1,984 returns) — Buyer remorse and impulse purchases.
3. **Product Damaged:** **16.01%** (1,933 returns) — Transit/packaging failure.
4. **Quality Issue:** **14.01%** (1,692 returns) — Fabric/material mismatch vs. listing.
5. **Product Not as Expected:** **11.94%** (1,442 returns).
6. **Wrong Product Shipped:** **11.55%** (1,394 returns) — Warehouse picking error.
7. **Late Delivery:** **7.95%** (960 returns).
8. **Other:** **3.88%** (469 returns).

### B. Financial Impact
* **Total Value Refunded:** **INR 105,898,381.80** (~105.9 Million INR).
* **Average Refund per Return:** **INR 10,316.45**.
* **Processing Status:** 70% Refund Completed, 15% Approved, 8% Rejected, 7% Pending.

---

## 4. Key Behavioral & Predictor Findings

1. **Product Category is the Strongest Domain Differentiator:**
   * **Footwear** has the highest return rate at **8.60%** (1,205 returns / 14,012 orders).
   * **Fashion** accounts for the massive majority of total return volume with **7.52%** (3,185 returns / 42,332 orders).
   * **Watches** (5.42%), **Bags** (5.39%), and **Furniture** (5.23%) have moderate return rates.
   * **Toys** (4.50%) and **Groceries** exhibit the lowest return propensity.
2. **Basket Size & Multi-Item Orders:**
   * `total_items` shows the highest positive linear correlation with returns ($r = +0.036$). Larger baskets mechanically increase the probability that at least one item is rejected.
3. **Logistics & Delivery Delays:**
   * Average transit time is **4.10 days** (range: 0 to 14 days).
   * **Delayed Shipments** exhibit higher return propensity (**10.45%**) compared to on-time deliveries (**10.08%**).
4. **Customer Demographics:**
   * **Customer Segments:** Premium (9.94%), Regular (9.46%), Budget (9.20%), New (8.06%).
   * **Age Brackets:** 45–54 age cohort has the highest return rate (9.88%), while customers aged 55+ have the lowest (8.46%).

---

## 5. Critical Data Leakage & Feature Selection Directives

> [!CAUTION]
> **Data Leakage Warnings for Model Pipeline:**
> 1. **`order_status`**: Out of 6,681 orders with `order_status == 'Returned'`, 100% are returns. Using `order_status` will cause **100% artificial accuracy and catastrophic production failure**. **Must be dropped**.
> 2. **`returns.csv` Attributes**: `return_reason`, `refund_amount`, `return_date`, and `return_status` are created *after* the return event occurs. Never pass them into the feature matrix $X$.
> 3. **`products.return_rate_baseline`**: This is the underlying simulation seed parameter. Using it would introduce synthetic leakage not available in real-world environments.
> 4. **`customer_reviews`**: Written post-delivery; must be excluded if predicting return at checkout time.
> 5. **Identifiers to Purge:** `order_id`, `customer_id`, `product_id`, `campaign_id`, `order_date`, `order_time`.

---

## 6. Machine Learning Suitability Scorecard & Final Verdict

| Dimension | Score / Status | Actionable Implementation Strategy |
| :--- | :---: | :--- |
| **Statistical Capacity** | **Exceptional** | 100,000 orders provides massive statistical power for gradient boosted decision trees. |
| **Class Imbalance** | **9.52 : 1 Imbalance** | Use cost-sensitive weighting (`scale_pos_weight = 9.52`) or SMOTE + Stratified 5-Fold CV. |
| **Feature Engineering** | **High Potential** | Engineer customer historical return rate, basket item count, and category return propensity. |
| **Algorithm Selection** | **Recommended** | **LightGBM**, **XGBoost**, and **CatBoost** (handles high-cardinality Indian states/cities). |
| **Evaluation Threshold** | **Tuned Threshold** | Optimize decision threshold (typically between $P \approx 0.20$ and $0.30$) to maximize F1/Recall. |

### Final Recommendation
**APPROVED & HIGHLY RECOMMENDED FOR ML MODELING**.  
Dataset 2 is relationally richer, larger, and more operationally complete than Dataset 1. Once data leakage columns are dropped, it is ready for production model training.
