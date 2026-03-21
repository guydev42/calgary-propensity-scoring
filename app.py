"""Streamlit dashboard for customer propensity and upsell scoring."""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

PROJECT_DIR = os.path.dirname(__file__)
sys.path.insert(0, PROJECT_DIR)

from src.data_loader import load_data, engineer_features, get_feature_columns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Propensity scoring", layout="wide")

UPSELL_VALUE = 15.0


@st.cache_data
def get_data():
    df = load_data()
    df = engineer_features(df)
    return df


@st.cache_resource
def get_model():
    model_path = os.path.join(PROJECT_DIR, "models", "best_model.pkl")
    scaler_path = os.path.join(PROJECT_DIR, "models", "scaler.pkl")
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return joblib.load(model_path), joblib.load(scaler_path)
    return None, None


@st.cache_data
def get_decile_data():
    path = os.path.join(PROJECT_DIR, "models", "decile_analysis.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Segmentation", "Model", "Scorer"],
)

df = get_data()
model, scaler = get_model()


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Customer propensity scoring dashboard")
    st.markdown("Campaign performance metrics and response analysis.")

    col1, col2, col3, col4 = st.columns(4)
    total = len(df)
    responders = df["responded"].sum()
    rate = responders / total

    col1.metric("Total customers", f"{total:,}")
    col2.metric("Responders", f"{responders:,}")
    col3.metric("Response rate", f"{rate:.1%}")
    col4.metric("Potential upsell revenue",
                f"${responders * UPSELL_VALUE * 12:,.0f}/yr")

    st.subheader("Response distribution")
    fig = px.histogram(df, x="responded", color="responded",
                       color_discrete_map={0: "#90CAF9", 1: "#1565C0"},
                       labels={"responded": "Responded"},
                       title="Campaign response distribution")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key statistics")
    stats = df.describe().round(2)
    st.dataframe(stats, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Segmentation
# ---------------------------------------------------------------------------
elif page == "Segmentation":
    st.title("Customer segmentation analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["By plan", "By tenure", "By income", "By channel"])

    with tab1:
        plan_resp = (
            df.groupby("current_plan")["responded"]
            .agg(["mean", "count", "sum"])
            .reset_index()
        )
        plan_resp.columns = ["Plan", "Response rate", "Total", "Responders"]
        plan_resp = plan_resp.set_index("Plan").loc[["Basic", "Standard", "Premium"]].reset_index()

        fig = px.bar(plan_resp, x="Plan", y="Response rate",
                     text=plan_resp["Response rate"].apply(lambda x: f"{x:.1%}"),
                     color="Plan",
                     color_discrete_map={"Basic": "#90CAF9", "Standard": "#2196F3", "Premium": "#0D47A1"},
                     title="Response rate by current plan")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(plan_resp, use_container_width=True)

    with tab2:
        df["tenure_bucket"] = pd.cut(
            df["tenure_months"], bins=[0, 12, 24, 48, 72, 120],
            labels=["0-12", "13-24", "25-48", "49-72", "73-120"]
        )
        tenure_resp = df.groupby("tenure_bucket")["responded"].mean().reset_index()
        tenure_resp.columns = ["Tenure bucket", "Response rate"]

        fig = px.bar(tenure_resp, x="Tenure bucket", y="Response rate",
                     text=tenure_resp["Response rate"].apply(lambda x: f"{x:.1%}"),
                     title="Response rate by tenure bucket",
                     color_discrete_sequence=["#2196F3"])
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        df["income_bucket"] = pd.qcut(df["income"], 5, labels=[
            "Bottom 20%", "20-40%", "40-60%", "60-80%", "Top 20%"
        ])
        inc_resp = df.groupby("income_bucket")["responded"].mean().reset_index()
        inc_resp.columns = ["Income quintile", "Response rate"]

        fig = px.bar(inc_resp, x="Income quintile", y="Response rate",
                     text=inc_resp["Response rate"].apply(lambda x: f"{x:.1%}"),
                     title="Response rate by income quintile",
                     color_discrete_sequence=["#2196F3"])
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        ch_resp = df.groupby("channel_preference")["responded"].mean().reset_index()
        ch_resp.columns = ["Channel", "Response rate"]
        ch_resp = ch_resp.sort_values("Response rate", ascending=False)

        fig = px.bar(ch_resp, x="Channel", y="Response rate",
                     text=ch_resp["Response rate"].apply(lambda x: f"{x:.1%}"),
                     title="Response rate by channel preference",
                     color_discrete_sequence=["#2196F3"])
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Model
# ---------------------------------------------------------------------------
elif page == "Model":
    st.title("Model performance and decile analysis")

    decile_df = get_decile_data()

    if decile_df is None:
        st.warning("Run the model pipeline first (src/model.py) to generate results.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Lift chart", "Decile analysis", "Campaign ROI"])

    with tab1:
        st.subheader("Cumulative lift chart")
        pcts = np.arange(1, len(decile_df) + 1) / len(decile_df) * 100
        cum_resp = decile_df["cumulative_response_pct"].values * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pcts, y=cum_resp, mode="lines+markers",
            name="Model", line=dict(color="#2196F3", width=3)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 100], y=[0, 100], mode="lines",
            name="Random", line=dict(color="gray", dash="dash")
        ))
        fig.update_layout(
            xaxis_title="Percentage of customers contacted",
            yaxis_title="Cumulative responders captured (%)",
            title="Lift chart",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Lift at top 30%
        top30_capture = cum_resp[2] if len(cum_resp) > 2 else cum_resp[-1]
        st.info(f"By contacting the top 30% of customers, we capture "
                f"{top30_capture:.1f}% of all responders.")

    with tab2:
        st.subheader("Decile analysis table")
        display_df = decile_df.copy()
        display_df["response_rate"] = display_df["response_rate"].apply(lambda x: f"{x:.2%}")
        display_df["cumulative_response_pct"] = display_df["cumulative_response_pct"].apply(lambda x: f"{x:.1%}")
        display_df["expected_revenue"] = display_df["expected_revenue"].apply(lambda x: f"${x:,.2f}")
        st.dataframe(display_df, use_container_width=True)

        fig = px.bar(decile_df, x="decile", y="response_rate",
                     title="Response rate by propensity decile",
                     labels={"decile": "Decile (0 = highest)", "response_rate": "Response rate"},
                     color_discrete_sequence=["#2196F3"])
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Campaign ROI comparison")

        total_cust = decile_df["n_customers"].sum()
        total_resp = decile_df["n_responders"].sum()
        overall_rate = total_resp / total_cust

        top3 = decile_df[decile_df["decile"] <= 2]
        tgt_cust = top3["n_customers"].sum()
        tgt_resp = top3["n_responders"].sum()
        tgt_rate = tgt_resp / tgt_cust

        cost_per_contact = st.slider("Cost per contact ($)", 0.5, 10.0, 2.0, 0.5)

        mass_cost = total_cust * cost_per_contact
        mass_rev = total_resp * UPSELL_VALUE * 12
        tgt_cost = tgt_cust * cost_per_contact
        tgt_rev = tgt_resp * UPSELL_VALUE * 12

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Mass campaign")
            st.metric("Contacts", f"{total_cust:,}")
            st.metric("Expected responders", f"{total_resp:,}")
            st.metric("Cost", f"${mass_cost:,.0f}")
            st.metric("Annual revenue", f"${mass_rev:,.0f}")
            mass_roi = (mass_rev - mass_cost) / mass_cost * 100
            st.metric("ROI", f"{mass_roi:.0f}%")

        with col2:
            st.markdown("### Targeted (top 3 deciles)")
            st.metric("Contacts", f"{tgt_cust:,}")
            st.metric("Expected responders", f"{tgt_resp:,}")
            st.metric("Cost", f"${tgt_cost:,.0f}")
            st.metric("Annual revenue", f"${tgt_rev:,.0f}")
            tgt_roi = (tgt_rev - tgt_cost) / tgt_cost * 100
            st.metric("ROI", f"{tgt_roi:.0f}%")

        st.success(
            f"Targeting top 3 deciles saves ${mass_cost - tgt_cost:,.0f} in contact costs "
            f"while capturing {tgt_resp}/{total_resp} responders. "
            f"Conversion lift: {(tgt_rate / overall_rate - 1) * 100:.0f}%."
        )


# ---------------------------------------------------------------------------
# Page: Scorer
# ---------------------------------------------------------------------------
elif page == "Scorer":
    st.title("Individual customer scorer")
    st.markdown("Input customer features to predict upsell probability.")

    if model is None:
        st.warning("Run the model pipeline first (src/model.py) to load the model.")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 80, 35)
        income = st.number_input("Annual income ($)", 25000, 200000, 65000, step=5000)
        tenure = st.number_input("Tenure (months)", 1, 120, 24)
        plan = st.selectbox("Current plan", ["Basic", "Standard", "Premium"])
        monthly = st.number_input("Monthly spend ($)", 20.0, 180.0, 65.0, step=5.0)

    with col2:
        data_gb = st.number_input("Data usage (GB)", 0.5, 80.0, 10.0, step=1.0)
        calls = st.number_input("Call minutes", 50, 3000, 300, step=50)
        sms = st.number_input("SMS count", 0, 500, 80, step=10)
        streaming = st.checkbox("Has streaming")
        international = st.checkbox("Has international")

    with col3:
        insurance = st.checkbox("Has device insurance")
        prev_response = st.checkbox("Previous upsell response")
        channel = st.selectbox("Channel preference",
                               ["Email", "SMS", "App notification", "Direct mail"])

    if st.button("Predict propensity", type="primary"):
        plan_map = {"Basic": 0, "Standard": 1, "Premium": 2}
        channel_map = {"Email": 0, "SMS": 1, "App notification": 2, "Direct mail": 3}

        plan_enc = plan_map[plan]
        channel_enc = channel_map[channel]
        revenue_per_tenure = monthly / max(tenure, 1)

        # Usage intensity (approximate with raw values centered)
        data_norm = (data_gb - 15.0) / 12.0
        calls_norm = (calls - 350) / 250
        sms_norm = (sms - 80) / 40
        usage_int = (data_norm + calls_norm + sms_norm) / 3

        service_ct = int(streaming) + int(international) + int(insurance)
        upsell_room = 2 - plan_enc
        income_x_tenure = income * tenure / 1e6
        data_x_stream = data_gb * int(streaming)

        features = np.array([[
            age, income, tenure, monthly, data_gb, calls, sms,
            int(streaming), int(international), int(insurance),
            int(prev_response), plan_enc, channel_enc,
            revenue_per_tenure, usage_int, service_ct,
            upsell_room, income_x_tenure, data_x_stream,
        ]])

        features_scaled = scaler.transform(features)
        prob = model.predict_proba(features_scaled)[0, 1]

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Upsell probability", f"{prob:.1%}")

        with col_b:
            if prob >= 0.3:
                st.success("HIGH propensity - prioritize for targeted campaign")
                action = "Send personalized upsell offer via " + channel.lower()
            elif prob >= 0.15:
                st.warning("MEDIUM propensity - include in secondary wave")
                action = "Include in batch campaign"
            else:
                st.info("LOW propensity - do not target")
                action = "Exclude from current campaign"
            st.markdown(f"**Recommended action:** {action}")

        expected_value = prob * UPSELL_VALUE * 12
        st.markdown(f"**Expected annual value:** ${expected_value:.2f}")
