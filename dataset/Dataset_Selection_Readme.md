# Dataset Evaluation & Selection Report: E-Commerce Return Prediction

**Project:** E-Commerce Return Prediction  
**Objective:** Compare candidate datasets through Exploratory Data Analysis (EDA) and justify the selection of Dataset 3 for predicting product returns.

---

## 1. Executive Summary

Our primary goal is to build a machine learning model capable of predicting whether an order will be returned at the time of purchase. 

To determine the most viable data foundation, we conducted Exploratory Data Analysis (EDA) across three candidate datasets. Rather than choosing based on convenience, we evaluated each dataset on fundamental statistical criteria:
- **Class separability and feature discriminative power**
- **Real-world behavioral distributions (category risk, price elasticity)**
- **Information integrity at checkout (leakage-free pre-delivery signal)**
- **Statistical power and sample representativeness**

Based on our EDA findings, **Dataset 3** emerged as the clearly superior candidate. Below is our comparative analysis and the empirical rationale behind our selection.

---

## 2. Comparative Analysis of Candidate Datasets

| Evaluation Dimension | Dataset 1 | Dataset 2 | Dataset 3 (Selected) |
| :--- | :--- | :--- | :--- |
| **Observation Scale** | 30,000 orders | 100,000 orders | 50,000,000 (Source) / 5,000,000 (Project Scope) |
| **Data Nature** | Small transactional sample | Synthetically simulated dataset | Authentic commercial distribution patterns |
| **Class Imbalance** | 10.11% Returned (~8.89 : 1) | 9.51% Returned (~9.52 : 1) | **11.30% Returned (~7.85 : 1)** |
| **Category Variance** | Moderate (8.7% to 13.9%) | Compressed (4.5% to 8.6%) | **Wide & Realistic (1.97% to 22.19%)** |
| **Price Sensitivity Signal** | Weak / Uncorrelated | Moderate | **Strong & Systematically Discriminative** |
| **Missing Values** | 0% | High structural missingness (e.g. 82.8% in campaigns) | **0% (100% complete across all attributes)** |
| **Temporal Stability** | Cross-sectional sample | Multi-year simulation | **24 Months Continuous (11.30% ± 0.4% monthly)** |

---

## 3. Detailed EDA Findings Across Candidate Datasets

### Dataset 1: Statistical Limitations & Signal Deficiency
Our EDA on Dataset 1 revealed significant limitations that make it unsuitable for training robust predictive models:

1. **Lack of Discriminative Power in Numerical Features:**
   * In a real-world setting, operational factors such as shipping duration often influence return propensity. In Dataset 1, however, `Delivery_Days` showed virtually identical distributions for kept orders and returned orders (mean of 4.44 days for both, with a near-zero correlation of $r \approx 0.003$).
   * Similarly, `Discount_Percent` exhibited no meaningful variance across classes, failing to provide predictive separation.
2. **Homogeneous Customer Demographics:**
   * Return rates across customer tiers were essentially flat (New: 11.53%, Regular: 9.44%, Premium: 9.32%). Customer age groups showed almost no statistical variance in return behavior, depriving the model of demographic signals.
3. **High Multicollinearity & Redundant Information:**
   * Several numerical features exhibited extreme collinearity (e.g., `Order_Amount` and `Tax_Amount` had a correlation of 0.93), adding feature redundancy without adding new behavioral information.
4. **Sample Size Constraints for Minority Class:**
   * With only 30,000 total observations and a 10.11% return rate, the dataset contains just **3,033 positive return cases**. When split into training, validation, and testing sets, the minority class becomes too sparse for gradient boosting algorithms to reliably discover multi-variable interaction patterns without overfitting.
5. **Checkout Information Gap:**
   * After removing target-leaking post-fulfillment columns (such as `Order_Status` and post-delivery `Review_Rating`), the remaining features lacked sufficient variance to distinguish likely returns from kept orders.

---

### Dataset 2: Synthetic Artifacts & Compressed Distribution Range
While Dataset 2 contains a larger volume of 100,000 orders, our EDA identified substantial distributional issues stemming from its synthetic generation:

1. **Synthetic Generation Artifacts:**
   * Dataset 2 is generated via simulation scripts using rule-based generators. This introduces synthetic uniformity and artificial parameters (such as a baseline return rate seed in the product catalog) that do not reflect organic consumer shopping behavior.
2. **Compressed Category Dispersion (Unrealistic Domain Distribution):**
   * In actual e-commerce markets, product categories exhibit drastic differences in return behavior. Apparel and footwear typically experience return rates exceeding 20% due to tactile, sizing, and styling uncertainty, whereas groceries and books remain under 4%.
   * In Dataset 2, category return rates were artificially compressed into a narrow range: Footwear at 8.60%, Fashion at 7.52%, Watches at 5.42%, Bags at 5.39%, and Toys at 4.50%. This flat distribution fails to mirror real-life e-commerce industry realities.
3. **Post-Event Bias vs. Checkout-Time Predictability:**
   * A large portion of Dataset 2's detailed fields describes what happens *after* a return is initiated (such as return reasons, refund amounts, and return processing statuses).
   * For a model predicting returns at or before delivery, post-event fields cannot be used without causing severe data leakage. Once these post-event fields are excluded, the pre-fulfillment features show limited predictive signal.

---

### Dataset 3: Realistic Distributions & Strong Empirical Signals
Dataset 3 proved to be the most aligned with actual e-commerce operations, demonstrating clear, non-linear relationships and high data integrity:

1. **Realistic, High-Variance Category Dynamics:**
   * The distribution of returns across categories in Dataset 3 closely mirrors real-world commercial benchmarks:
     * **Clothing:** **22.19% return rate** (Highest risk — captures real-life sizing and styling uncertainty)
     * **Electronics:** **13.75% return rate** (High risk — performance and specification mismatch)
     * **Sports:** **11.03% return rate**
     * **Home & Kitchen:** **8.74% return rate**
     * **Toys:** **8.06% return rate**
     * **Beauty:** **7.32% return rate**
     * **Books:** **3.96% return rate** (Low risk — standardized format)
     * **Grocery:** **1.97% return rate** (Minimal risk — non-perishable consumables)
   * This wide spread (from ~2% up to ~22%) gives machine learning models a powerful, realistic domain signal to segment high-risk from low-risk orders.
2. **Distinct Price Elasticity & Financial Distribution:**
   * Behavioral economics suggests that customers are far more inclined to return high-value purchases than low-cost impulse buys. Dataset 3 strongly reflects this real-life pattern:
     * **Unit Price:** Returned orders have a median of **\$65.52** and mean of **\$122.19**, compared to **\$49.60** median and **\$104.93** mean for delivered orders (+\$17.26 higher).
     * Low-cost items ($\le \$18.47$) exhibit only a **3.78%** return rate, whereas mid-to-high tier items (\$37–\$71) experience a **12.23%** return rate.
     * **Order Total:** Returned orders average **\$236.33**, compared to **\$203.40** for kept items (+\$32.93 higher).
3. **Realistic Commercial Shipping Thresholds:**
   * EDA revealed a clear, authentic business rule: orders under \$50 incur standard shipping fees, while orders over \$100 universally receive 100% free shipping. This mirrors common e-commerce policies that incentivize customers to add extra items to carts to unlock free shipping—a known driver of subsequent returns.
4. **Chronological Consistency Over 24 Months:**
   * Spanning from January 2024 through December 2025, monthly return rates remain remarkably stable (~11.30%, ranging between 10.90% and 11.73%). This ensures that validation splits and test evaluations are not contaminated by abrupt distribution shifts.
5. **Flawless Completeness & Record Integrity:**
   * 0% missing values across all features, 0 duplicate records, and exactly one row per transaction. This ensures the modeling pipeline trains on consistent, high-integrity data.

---

## 4. Computational Scale Justification: Why 5M Rows (and Not 50M)

While Dataset 3 originates from an open 50-Million-order repository, using the entire 50M records for a project is neither computationally feasible nor statistically necessary:

1. **Hardware & Computational Feasibility:**
   * Ingesting and processing 50 million records requires massive memory allocations and distributed computing infrastructure that exceeds standard workstation capabilities.
   * Training gradient boosted decision trees (LightGBM, XGBoost, CatBoost), running hyperparameter tuning, and performing cross-validation across 50M rows would result in excessive training times and out-of-memory errors.
2. **Statistical Sufficiency & Diminishing Returns:**
   * In statistical learning, error rates follow power-law convergence. Once a dataset reaches millions of well-distributed observations, marginal gains in model generalization become negligible.
   * A sample of **5,000,000 rows (50 Lakhs)** offers tremendous statistical depth—providing approximately 450,000 positive return cases—which is more than sufficient for high-capacity non-linear models.
3. **Stratified Splitting Strategy:**
   * To ensure the 5M dataset preserves the authentic properties of the broader population:
     * We filter the population down to finalized orders (`delivered` = 0 vs. `returned` = 1), removing non-final states (`shipped`, `processing`, `cancelled`).
     * We perform **Stratified Splitting** across both the target variable (`return_target`) and `product_category`.
     * This guarantees that the **~11.3% baseline return rate** and the category-specific risk proportions (e.g., 22.2% in clothing, 2% in groceries) are strictly preserved across both training and test sets.

---

## 5. Conclusion & Selection Verdict

Our choice of Dataset 3 is grounded in rigorous EDA evidence:
- **Dataset 1** suffers from a restricted sample size (only 3k returns), heavy collinearity, and flat distributions across key features.
- **Dataset 2** exhibits synthetic compression, where category-level return rates fail to reflect the wide variance observed in real consumer markets.
- **Dataset 3** provides authentic domain distributions: massive category separation (2% to 22%), clear price elasticity signals, verified free-shipping business rules, and high temporal stability.

By extracting a **5-Million-row stratified corpus**, we achieve an optimal balance: enterprise-grade statistical power and authentic real-world behavior, paired with computationally feasible training and validation for our project.
