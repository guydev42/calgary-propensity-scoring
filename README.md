<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Propensity%20%26%20Upsell%20Scoring&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Rank%20customers%20by%20upsell%20likelihood%20to%20maximize%20campaign%20ROI&descAlignY=55&descSize=16" width="100%"/>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Random%20Forest-AUC%200.775-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/ROI-2%2C155%25-f59e0b?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-f59e0b?style=for-the-badge"/>
</p>

<p>
  <a href="#overview">Overview</a> •
  <a href="#key-results">Key results</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#methodology">Methodology</a>
</p>

</div>

---

## Overview

> **A propensity scoring system that ranks 8,000 telecom customers by upsell likelihood, achieving 2,155% campaign ROI by targeting the top deciles.**

Telecom companies run upsell campaigns to move customers from lower-tier to higher-tier plans, but mass campaigns are expensive and yield low conversion rates. This project builds a propensity scoring system that ranks customers by their likelihood to respond, enabling marketing teams to focus resources on the most promising prospects. Using calibrated probability estimates and decile analysis, the model identifies the top 30% of customers who capture 63% of all responders, dramatically improving campaign ROI compared to untargeted outreach.

```
Problem   →  Mass upsell campaigns are costly with low conversion rates
Solution  →  Calibrated Random Forest scores customers by response probability
Impact    →  Top decile hits 37.7% response rate, targeted campaign yields 2,155% ROI
```

---

## Key results

| Metric | Value |
|--------|-------|
| Best model | Random Forest (AUC 0.775) |
| Top decile response rate | 37.7% vs 1.3% baseline |
| Top 3 deciles capture | 63.2% of all responders |
| Targeted campaign ROI | 2,155% |
| Customers scored | 8,000 |

---

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Raw data    │───▶│  Feature         │───▶│  Model training     │
│  (8K rows)   │    │  engineering     │    │  (LR, RF, XGB)      │
└─────────────┘    └──────────────────┘    └──────────┬──────────┘
                                                      │
                          ┌───────────────────────────┘
                          ▼
              ┌──────────────────────┐    ┌──────────────────────┐
              │  Probability         │───▶│  Decile              │
              │  calibration         │    │  analysis            │
              └──────────────────────┘    └──────────┬───────────┘
                                                     │
                          ┌──────────────────────────┘
                          ▼
              ┌──────────────────────┐    ┌──────────────────────┐
              │  Campaign ROI        │───▶│  Streamlit app       │
              │  comparison          │    │  (dashboard)         │
              └──────────────────────┘    └──────────────────────┘
```

<details>
<summary><b>Project structure</b></summary>

```
project_15_propensity_upsell_scoring/
├── data/
│   ├── marketing_campaign.csv       # 8,000 customer records
│   └── generate_data.py             # Synthetic data generator
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Data loading and preprocessing
│   └── model.py                     # Training, calibration, evaluation
├── notebooks/
│   └── 01_eda.ipynb                 # Exploratory data analysis
├── models/
│   ├── best_model.pkl               # Serialized Random Forest
│   ├── scaler.pkl                   # Feature scaler
│   └── decile_analysis.csv          # Decile-level metrics
├── figures/
│   ├── lift_chart.png               # Cumulative lift curve
│   ├── decile_response.png          # Response rate by decile
│   ├── calibration_curves.png       # Pre/post calibration
│   └── feature_importance.png       # Top feature rankings
├── app.py                           # Streamlit dashboard
├── requirements.txt
└── README.md
```

</details>

---

## Quickstart

```bash
# Clone and navigate
git clone https://github.com/guydev42/calgary-data-portfolio.git
cd calgary-data-portfolio/project_15_propensity_upsell_scoring

# Install dependencies
pip install -r requirements.txt

# Generate dataset
python data/generate_data.py

# Train models and generate outputs
python -m src.model

# Launch dashboard
streamlit run app.py
```

---

## Dataset

| Property | Details |
|----------|---------|
| Source | Synthetic telecom marketing data |
| Records | 8,000 customers |
| Features | Demographics, usage patterns, service subscriptions, campaign history |
| Target | Campaign response (binary, ~12% response rate) |
| Class balance | Imbalanced; handled with class weighting |

---

## Tech stack

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-189FDD?style=for-the-badge&logo=xgboost&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white"/>
</p>

---

## Methodology

<details>
<summary><b>Feature engineering</b></summary>

- Revenue per tenure month
- Usage intensity composite score
- Service subscription count
- Upsell headroom (gap between current and max tier)
- Interaction terms: income x tenure, data usage x streaming
</details>

<details>
<summary><b>Model training</b></summary>

- Logistic Regression, Random Forest, and XGBoost with class balancing
- Cross-validated hyperparameter tuning
- Evaluation on AUC, precision-recall, and lift metrics
</details>

<details>
<summary><b>Probability calibration</b></summary>

- Isotonic regression via `CalibratedClassifierCV`
- Produces reliable probability estimates for ranking and decision-making
- Calibration curves validated pre- and post-calibration
</details>

<details>
<summary><b>Decile analysis and campaign ROI</b></summary>

- Customers ranked into 10 groups by predicted probability
- Response rate and cumulative capture calculated per decile
- Targeted campaign (top 3 deciles) vs mass campaign cost-benefit comparison
- 2,155% ROI achieved by focusing on highest-propensity segments
</details>

---

## Acknowledgements

Built as part of the [Calgary Data Portfolio](https://guydev42.github.io/calgary-data-portfolio/).

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

**[Ola G](https://github.com/guydev42)**
</div>
