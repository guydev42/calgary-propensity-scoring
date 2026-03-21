# Propensity and upsell scoring

## Problem statement

Telecom companies run upsell campaigns to move customers from lower-tier to higher-tier plans, but mass campaigns are expensive and yield low conversion rates. This project builds a propensity scoring system that ranks customers by their likelihood to respond, enabling marketing teams to focus resources on the most promising prospects and dramatically improve campaign ROI.

## Approach

- **Data**: 8,000 synthetic customer records with demographics, usage patterns, service subscriptions, and campaign response history (~12% response rate)
- **Feature engineering**: revenue per tenure, usage intensity composite, service count, upsell headroom, and interaction terms (income x tenure, data x streaming)
- **Models**: Logistic Regression, Random Forest, and XGBoost trained with class balancing
- **Calibration**: isotonic regression via `CalibratedClassifierCV` to produce reliable probability estimates
- **Decile analysis**: customers ranked into 10 groups by predicted probability, with response rate and cumulative capture calculated per decile
- **Campaign ROI**: targeted (top 3 deciles) vs mass campaign cost-benefit comparison

## Key results

| Metric | Value |
|--------|-------|
| Best model | Random Forest (AUC 0.775) |
| Top decile response rate | 37.7% vs 1.3% baseline |
| Top 3 deciles capture | 63.2% of all responders |
| Targeted campaign ROI | 2,155% |

## How to run

```bash
pip install -r requirements.txt
python data/generate_data.py
python -m src.model
streamlit run app.py
```

## Project structure

```
project_15_propensity_upsell_scoring/
  data/
    marketing_campaign.csv
    generate_data.py
  src/
    __init__.py
    data_loader.py
    model.py
  notebooks/
    01_eda.ipynb
  models/
    best_model.pkl
    scaler.pkl
    decile_analysis.csv
  figures/
    lift_chart.png
    decile_response.png
    calibration_curves.png
    feature_importance.png
  app.py
  requirements.txt
  README.md
```

## Technical stack

Python, scikit-learn, XGBoost, pandas, matplotlib, Streamlit
