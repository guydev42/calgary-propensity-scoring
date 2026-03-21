"""Generate synthetic marketing campaign data for propensity scoring."""

import numpy as np
import pandas as pd
import os

def generate_marketing_campaign_data(n=8000, seed=42):
    """Generate 8,000 customer records with realistic upsell response patterns."""
    rng = np.random.RandomState(seed)

    customer_ids = [f"CUST_{i:06d}" for i in range(1, n + 1)]

    age = rng.normal(42, 12, n).clip(18, 80).astype(int)
    income = rng.lognormal(10.8, 0.45, n).clip(25000, 200000).astype(int)
    tenure_months = rng.exponential(30, n).clip(1, 120).astype(int)

    plan_probs = [0.35, 0.45, 0.20]
    current_plan = rng.choice(["Basic", "Standard", "Premium"], n, p=plan_probs)

    plan_base = {"Basic": 45, "Standard": 70, "Premium": 110}
    monthly_spend = np.array([
        rng.normal(plan_base[p], 10) for p in current_plan
    ]).clip(20, 180).round(2)

    data_usage_gb = (
        rng.lognormal(2.0, 0.7, n) + (income / 50000) * 2
    ).clip(0.5, 80).round(2)

    call_minutes = rng.lognormal(5.5, 0.6, n).clip(50, 3000).round(0).astype(int)
    sms_count = rng.poisson(80, n).clip(0, 500)

    has_streaming = rng.binomial(1, 0.40, n)
    has_international = rng.binomial(1, 0.15, n)
    has_device_insurance = rng.binomial(1, 0.30, n)

    previous_upsell_response = rng.binomial(1, 0.18, n)

    channel_probs = [0.35, 0.25, 0.25, 0.15]
    channel_preference = rng.choice(
        ["Email", "SMS", "App notification", "Direct mail"],
        n, p=channel_probs
    )

    # Build response probability with realistic correlations
    plan_map = {"Basic": 0, "Standard": 1, "Premium": 2}
    plan_numeric = np.array([plan_map[p] for p in current_plan])

    logit = (
        -3.15
        + 0.6 * ((income - 50000) / 40000)               # higher income -> more likely
        + 0.5 * ((tenure_months - 20) / 25)                # longer tenure -> more likely
        + 0.8 * (plan_numeric == 1).astype(float)          # Standard plan most upsellable
        - 0.6 * (plan_numeric == 2).astype(float)          # Premium already at top
        + 0.4 * ((data_usage_gb - 10) / 15)                # high data usage
        + 1.0 * previous_upsell_response                   # previous responders
        + 0.2 * has_streaming                               # engaged customers
        - 0.1 * ((age - 42) / 12)                          # slight age effect
        + rng.normal(0, 0.3, n)                            # noise
    )
    prob = 1 / (1 + np.exp(-logit))
    responded = (rng.random(n) < prob).astype(int)

    # Verify approximate 12% response rate, adjust intercept if needed
    actual_rate = responded.mean()
    print(f"Response rate: {actual_rate:.3f} ({responded.sum()} / {n})")

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": age,
        "income": income,
        "tenure_months": tenure_months,
        "current_plan": current_plan,
        "monthly_spend": monthly_spend,
        "data_usage_gb": data_usage_gb,
        "call_minutes": call_minutes,
        "sms_count": sms_count,
        "has_streaming": has_streaming,
        "has_international": has_international,
        "has_device_insurance": has_device_insurance,
        "previous_upsell_response": previous_upsell_response,
        "channel_preference": channel_preference,
        "responded": responded,
    })

    out_path = os.path.join(os.path.dirname(__file__), "marketing_campaign.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} records to {out_path}")
    return df


if __name__ == "__main__":
    generate_marketing_campaign_data()
