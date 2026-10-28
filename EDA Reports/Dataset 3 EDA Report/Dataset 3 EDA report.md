# EDA Executive Report: E-Commerce Return Prediction
**Dataset:** `Dataset 3` (`ziadatalabs/FreeEcommerceOrders50M` via Hugging Face)  
**Notebook Analyzed:** [`Dataset_3_EDA.ipynb`]
**Sample Scale:** 500,000 Order Records (Spanning 2 Full Years: 2024-01-01 to 2025-12-30)  
**Task:** Binary Classification — Predict Customer Return Propensity (`Returned` vs. `Delivered`)

---

## 1. Dataset Scale & Structural Health

| Dimension | Metric / Observation | Engineering & ML Implication |
| :--- | :--- | :--- |
| **Total Sample Size** | **500,000 orders × 12 features** | Massive statistical power; ideal for deep decision trees, LightGBM, and XGBoost. |
| **Unit of Observation** | **Exactly 1 Row per Order** (100% 1-to-1) | No multi-line order aggregation required; clean transactional grain. |
| **Data Completeness** | **0% Missing Values (100% Complete)** | Zero missing entries across all 12 raw columns; no imputation pipelines needed. |
| **Record Integrity** | **0 Duplicate Rows / 0 Duplicate Order IDs** | Flawless primary key integrity. |
| **Temporal Coverage** | **24 Months** (2024-01-01 to 2025-12-30) | Clean two-year chronological horizon; perfect for temporal train-test splitting (Train 2024, Test 2025). |
| **Customer Breadth** | **392,155 Unique Customers** | Only 1.02 orders/customer; model must learn generalizable product/pricing signals rather than customer history. |
| **Catalog Breadth** | **244,914 Unique Product IDs** | Average 1.63 orders/SKU; individual product IDs should be omitted in favor of category & pricing features. |

---

## 2. Methodological Justification for 500k Sampling (vs. Full Multi-Million Dataset)

While the full underlying source dataset spans tens of millions of records (50M rows), a statistically representative sample of **500,000 observations** was utilized for this exploratory data analysis. Evaluating a 500k sample rather than the full multi-million-row corpus is justified by four key data visualization and computational principles:

1. **Overplotting:** In scatterplots and other dense visualizations, millions of observations overlap, hiding patterns and making individual points difficult to distinguish.
2. **Rendering Time:** Plotting libraries such as Matplotlib and Seaborn may take longer to render millions of points, slowing exploratory analysis.
3. **Memory Consumption:** Creating plot objects and intermediate arrays for large datasets can consume substantial RAM, increasing the risk of memory issues.
4. **Limited Visual Benefit:** For many plots, such as boxplots and histograms, plotting every individual observation does not necessarily provide more useful insight than using a representative sample or aggregated statistics.

---

## 3. Order Status Breakdown & Target Definition

Across the entire 500,000 transaction sample:
* **Delivered:** 354,803 (**70.96%**)
* **Shipped (In Transit):** 50,029 (**10.01%**)
* **Returned:** 45,209 (**9.04%**)
* **Processing:** 30,064 (**6.01%**)
* **Cancelled:** 19,895 (**3.98%**)

### Final Target Population for Return Prediction
For supervised classification, the valid population consists of finalized orders with known return outcomes (**Delivered + Returned = 400,012 orders**):

```
Target Distribution (Delivered vs. Returned):
├── Delivered (Kept): 354,803 orders (88.70%)
└── Returned:          45,209 orders (11.30%)
```

* **Class Imbalance Ratio:** **7.85 : 1** (~11.3% return rate).
* **Consistency:** Closely aligns with industry norms and earlier datasets (Dataset 1: 10.11%, Dataset 2: 9.51%).
* **Metric Warning:** **Do NOT evaluate using standard Accuracy** (an 88.7% baseline accuracy produces 0% return recall).
* **Recommended Metrics:** **PR-AUC (Average Precision)**, **ROC-AUC**, and **F1 / Recall at optimized decision thresholds**.

---

## 4. Key Feature Insights & Behavioral Drivers of Returns

### A. Product Category is the Single Strongest Predictive Driver
Observed return rates vary massively across product departments (from ~2% to >22%):

| Product Category | Orders in Scope | Returns | Return Rate (%) | Risk Level |
| :--- | :---: | :---: | :---: | :---: |
| **Clothing** | 87,834 | 19,490 | **22.19%** | **Extreme Risk** (~1 in 4 returned) |
| **Electronics** | 63,793 | 8,771 | **13.75%** | **High Risk** |
| **Sports** | 36,072 | 3,977 | **11.03%** | **Medium Risk** |
| **Home** | 56,402 | 4,930 | **8.74%** | **Moderate Risk** |
| **Toys** | 32,167 | 2,593 | **8.06%** | **Moderate Risk** |
| **Beauty** | 40,013 | 2,928 | **7.32%** | **Low-Medium Risk** |
| **Books** | 43,718 | 1,731 | **3.96%** | **Low Risk** |
| **Grocery** | 40,013 | 789 | **1.97%** | **Minimal Risk** |

### B. Price Sensitivity & Order Value Patterns
Returned orders exhibit systematically higher price points compared to delivered orders:
* **Unit Price:**
  * Delivered orders: Median = **\$49.60**, Mean = **\$104.93**
  * Returned orders: Median = **\$65.52**, Mean = **\$122.19** (**+\$17.26 higher**)
  * Low-cost items ($\le \$18.47$) have only a **3.78%** return rate, whereas mid/high-tier items (\$37–\$71) experience a **12.23%** return rate.
* **Order Total:**
  * Delivered orders: Median = **\$79.78**, Mean = **\$203.40**
  * Returned orders: Median = **\$106.18**, Mean = **\$236.33** (**+\$32.93 higher**)
  * Sub-\$35 orders return at only **4.64%**, while orders above \$60 return at **10.6%–10.8%**.

### C. The \$50 Free Shipping Policy Discovery
* **Prevalence:** **64.65%** of all orders (323,252 transactions) had **\$0.00 shipping cost**.
* **Policy Threshold:** The data reveals a strict universal business rule:
  * Orders $\le \$50$: **0.00% free shipping** (median shipping fee: \$6.50–\$7.25).
  * Orders \$50–\$100: **70%–93% free shipping**.
  * Orders $> \$100$: **100.00% free shipping** across every single product category.
* **Accounting Reconciliation:** `order_total = (quantity * unit_price) + shipping_cost` with a Pearson correlation of **0.99996**.
* **Return Impact:** Because returned items are higher-priced, they qualify for free shipping more frequently, resulting in a lower average shipping cost on returned items (\$1.71 vs. \$2.56).

### D. Payment Methods & Discounts (Weak/Neutral Signal)
* **Payment Method:** Near-uniform return rates across all payment rails:
  * `bnpl`: 11.56% | `debit_card`: 11.37% | `credit_card`: 11.28% | `gift_card`: 11.24% | `paypal`: 11.18%.
* **Discounts Applied:** Negligible impact (+0.48%):
  * Discount = True: **11.67%** return rate.
  * Discount = False: **11.19%** return rate.
* **Quantity Ordered:** Uniform return rate across basket quantities 1 through 6 (~8.9%–9.4%).

---

## 5. Temporal Stability & Seasonality

Monthly return percentages across all 24 months are exceptionally stable:
* **Mean Monthly Return Rate:** **11.30%**
* **Minimum Month:** **10.90%** (August 2024)
* **Maximum Month:** **11.73%** (February 2024)
* **Yearly Breakdown:**
  * **2024:** Delivered: 71.02%, Returned: 9.03%, Shipped: 9.96%, Cancelled: 3.98%
  * **2025:** Delivered: 70.90%, Returned: 9.05%, Shipped: 10.05%, Cancelled: 3.98%
* **Conclusion:** The dataset exhibits stable underlying mechanics without abrupt distribution drift, making chronological validation highly reliable.

---

## 6. Data Leakage Precautions & Modeling Strategy

> [!CAUTION]
> **Data Leakage & Preprocessing Rules:**
> 1. **Filter Non-Finalized Orders:** Filter out `shipped`, `processing`, and `cancelled` orders before modeling. Train strictly on `delivered` ($y=0$) vs. `returned` ($y=1$).
> 2. **Drop `order_status`:** Do not include `order_status` as an input feature (target leakage).
> 3. **Purge High-Cardinality Sparse IDs:** Drop `order_id`, `customer_id`, and `product_id`. Customers average 1.02 orders and products average 1.63 orders, meaning memorizing IDs will overfit.
> 4. **Feature Interactions:** Engineer composite features like `is_clothing`, `price_bracket`, `has_free_shipping`, and temporal variables (`day_of_week`, `month`).

---

## 7. Machine Learning Suitability Scorecard & Final Verdict

| Criterion | Evaluation / Status | Implementation Strategy |
| :--- | :---: | :--- |
| **Statistical Capacity** | **Superior (500k Clean Records sample size)** | Exceptional volume for training gradient boosted trees without risk of sample starvation. |
| **Class Balance** | **Realistic (11.30% Returns)** | Implement cost-sensitive learning (`scale_pos_weight = 7.85`) or tuned classification threshold ($P \approx 0.20$). |
| **Predictive Signal** | **Strong Non-Linear Signal** | Category + Unit Price + Order Total provide high predictive power for separating return propensities. |
| **Data Cleanliness** | **Pristine (0% Missing)** | Completely ready for feature encoding pipelines (`OneHotEncoder` for categories). |
| **Validation Scheme** | **Temporal Split** | Train on 2024 (200k orders) and evaluate on 2025 (200k orders) to simulate real-world forward testing. |

### Final Recommendation
**APPROVED & STRONGLY RECOMMENDED FOR ML MODELING**.  
Dataset 3 provides the largest and cleanest dataset among all three evaluated corpora. Its predictable category variances (22.2% in clothing vs. 2.0% in grocery) and clean free-shipping pricing mechanics make it an exceptional candidate for training production-grade return prediction models.
