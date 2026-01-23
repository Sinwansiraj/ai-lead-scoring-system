# 🚀 AI-Driven Lead Quality Scoring System

**A production-ready machine learning system for B2B CRM lead prioritization**

---

## 📌 Project Overview

Sales teams often waste **60–70% of their time on low-quality leads** that never convert.  
This project builds an **AI-driven Lead Quality Scoring System** that predicts the probability of lead conversion and assigns a **Lead Quality Score (0–100)** with actionable sales recommendations.

The system helps sales teams:
- Focus on **high-value leads**
- Reduce missed opportunities
- Improve conversion rates and sales efficiency

---

## 🎯 Business Objective

Build a **supervised binary classification model** to:
- Predict whether a lead will convert
- Rank leads by quality
- Recommend next-best sales actions

---

## 📊 Success Metrics

### Business Metrics
- 🔼 50%+ increase in conversion rate on top-scored leads  
- ⏱️ 30–40% reduction in time-to-close  
- 📈 Improved sales rep productivity  

### Technical Metrics
- ROC-AUC > **0.85**
- High **recall** to minimize missed high-value leads
- Balanced **precision** to reduce wasted sales effort

> **Why not accuracy?**  
> The dataset is highly imbalanced (≈12% conversion rate), and false negatives are more costly than false positives.

---

## 🧾 Dataset Description

A **synthetic but realistic CRM dataset** is generated to simulate real-world B2B SaaS scenarios.

### Features
- Lead source (Website, Referral, LinkedIn Ads, etc.)
- Industry (SaaS, Fintech, Healthcare, etc.)
- Company size (SMB, Mid-Market, Enterprise)
- Region
- Website visits
- Email opens and clicks
- Demo requested
- Days since last interaction
- Sales follow-up count

### Target Variable
- `converted` → Binary (1 = Converted, 0 = Not Converted)
- Conversion rate maintained at ~12%

---

## 🧠 Feature Engineering

Business-driven features were engineered to improve predictive power and interpretability:

### Engineered Features
- **Engagement Score (0–100)**  
  Composite score from website visits, email opens, and clicks.
  
- **Recency Score**  
  Exponential decay based on days since last interaction.

- **Interaction Intensity**  
  Engagement frequency per active time period.

- **Sales Readiness Score**  
  Combines demo request, engagement, and recency.

- **Interaction Features**
  - Demo × Engagement (strong buying intent signal)

- **Outlier Handling**
  - Caps extreme values (e.g., bot-like website visits)

---

## ⚙️ Data Preprocessing

- Missing value handling (business-rule based)
- Label encoding for categorical variables
- Feature scaling using `StandardScaler`
- Stratified train-test split to preserve conversion rate

---

## 🤖 Model Architecture

### Baseline Model
**Logistic Regression**
- Interpretable and fast
- Handles class imbalance using `class_weight='balanced'`
- Used as a performance benchmark

### Production Model
**XGBoost Classifier**
- Captures non-linear relationships
- Excellent performance on tabular data
- Handles class imbalance using `scale_pos_weight`
- Provides feature importance for business interpretation

---

## 📈 Model Evaluation

Models are evaluated using:
- ROC-AUC
- Precision
- Recall
- F1-score
- Confusion Matrix
- Business impact metrics

### Business Interpretation
- **True Positives** → Correctly identified hot leads  
- **False Positives** → Wasted sales effort  
- **False Negatives** → Missed revenue opportunities  

---

## 🔍 Feature Importance

XGBoost feature importance is extracted to:
- Explain why leads are scored high
- Build trust with sales teams
- Support data-driven decision-making

Top features typically include:
- Demo requested
- Engagement score
- Company size
- Recency score
- Interaction intensity

---

## 🎯 Lead Quality Scoring Logic

### Score Conversion
- Model probability → **Lead Quality Score (0–100)**

### Lead Categories
| Score Range | Category | Action |
|------------|---------|--------|
| 80–100 | Hot | Immediate sales call |
| 50–79 | Warm | Nurture & demo |
| < 50 | Cold | Automated drip campaigns |

---

## 📌 Prescriptive Sales Actions

Each lead receives an actionable recommendation:
- *Immediate Call – Strike while hot*
- *Personalized Demo Invite*
- *Value-driven Nurture Campaign*
- *Re-qualification Survey*

This bridges the gap between **ML predictions and business execution**.

---

## 🔁 End-to-End Pipeline

1. Data generation
2. Feature engineering
3. Data preprocessing
4. Model training (baseline & production)
5. Model evaluation
6. Lead scoring
7. Priority ranking
8. Deployment recommendations

---

## 🚀 Deployment Strategy

### Architecture
- REST API using **FastAPI**
- Endpoint: `/api/v1/score-lead`
- Supports real-time and batch scoring

### CRM Integration
- Webhooks on lead creation/update
- Nightly batch re-scoring
- Scores pushed to CRM custom fields

### Monitoring
- Prediction distribution drift
- Conversion rate by score band
- Feature drift detection (PSI)

### Retraining
- Monthly retraining
- A/B testing
- Gradual rollout (10% → 50% → 100%)

---

## 📊 Expected Business Impact

- ✅ 50–70% increase in sales efficiency  
- ✅ 2× higher conversion on top-scored leads  
- ✅ Faster deal closures  
- ✅ Clear ROI within 3 months  

---

## 🧠 Key Learnings

- Handling imbalanced datasets effectively
- Translating ML outputs into business actions
- Building interpretable and scalable ML pipelines
- Designing production-ready ML systems

---


## 📂 How to Run

```bash
pip install -r requirements.txt
python lead_scoring_system.py
