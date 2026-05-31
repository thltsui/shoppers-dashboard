import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
from app.data_loaders import load_train_sample, load_metrics, load_feature_importance_ret, load_category_cycle_summary
from app.theme import headline, metric_tile, callout

def render():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    sample_df = load_train_sample()
    metrics = load_metrics()
    
    ret_sample = sample_df[sample_df["cohort"] == "retention"].copy()
    observed_repeat_rate = ret_sample["repeater_int"].mean() * 100
    
    st.header("Retention (Prior Buyers) Strategy")
    st.markdown("For shoppers who have **previously** bought the brand, the coupon is distributed to encourage retention and prevent churn.")
        
    st.markdown("---")
    st.subheader("1. Data Limitations & Causal Interpretation")
    
    st.markdown("""
    **Why we can only do Predictive Modeling (Not Causal Uplift):**
    For the retention cohort (shoppers who have bought the brand before), we cannot causally interpret `repeater=1` as "the coupon caused the purchase". This is because:
    
    1. **No Control Group:** Every customer in this dataset received a coupon. We don't know what their baseline repeat rate would have been *without* a coupon.
    2. **Unknown Observation Window:** We don't know the timeframe during which `repeater` was tracked (e.g., 30 days vs 90 days).
    3. **Category-Cycle Confounding:** Customers buy items based on natural depletion cycles (e.g., milk weekly vs detergent monthly). Without knowing the observation window, short-cycle items will naturally have higher observed repeat rates simply because their natural cycle fits inside the observation window.
    
    *Therefore, the models below predict "who is most likely to be observed buying again", NOT "who was causally persuaded by the coupon".*
    """)
    
    st.markdown("---")
    st.subheader("2. Category Cycle Confounding")
    st.markdown("*Categories with short purchase cycles naturally show higher repeat rates within the unknown observation window.*")
    
    cycle_df = load_category_cycle_summary()
    
    # Calculate Line of Best Fit
    z = np.polyfit(cycle_df["mean_cat_avg_cycle_days"], cycle_df["ret_repeat_rate"], 1)
    p = np.poly1d(z)
    cycle_df["trendline"] = p(cycle_df["mean_cat_avg_cycle_days"])
    
    fig1 = px.scatter(
        cycle_df,
        x="mean_cat_avg_cycle_days",
        y="ret_repeat_rate",
        size="n_shoppers",
        hover_name="offered_category",
        labels={
            "mean_cat_avg_cycle_days": "Average Category Cycle (Days)",
            "ret_repeat_rate": "Retention Repeat Rate"
        },
        color_discrete_sequence=["#ff7f0e"]
    )
    # Add trendline
    fig1.add_trace(go.Scatter(
        x=cycle_df["mean_cat_avg_cycle_days"],
        y=cycle_df["trendline"],
        mode="lines",
        line=dict(color="red", dash="dash"),
        name="Trend (Negative Correlation)"
    ))
    fig1.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("---")
    st.subheader("3. Feature Importance & Directional Impact")
    
    st.markdown("**Feature Importance (Magnitude)**")
    feat_img = data_dir.parent.parent / "shoppers-pipeline" / "figures" / "05_feature_importance_ens_ret.png"
    if feat_img.exists():
        st.image(str(feat_img), use_container_width=True)
    else:
        callout("Feature importance chart not found.", "warning")
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**SHAP Directional Impact**")
    
    from app.data_loaders import load_shap_values_ret, load_shap_data_ret
    import shap
    import matplotlib.pyplot as plt
    
    try:
        shap_v = load_shap_values_ret()
        shap_d = load_shap_data_ret()
        
        fig_shap = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_v.values, shap_d, show=False)
        st.pyplot(fig_shap)
        plt.close(fig_shap)
        
        st.info("""
        **Ideal Target Customer Profile (Data-Driven by SHAP):**
        By analyzing the directional relationship between feature magnitudes (red=high, blue=low) and their impact on prediction (positive SHAP = more likely to repeat), the highest propensity target customer is:
        
        1. **Premium Brand Affinity:** `brand_premium_index` is strongly positively correlated. Customers who are historically willing to pay a high premium for the brand are highly likely to repeat.
        2. **Recent Absolute Brand Spend (Positive):** High absolute spending on the brand recently (`brand_prior_spend_30d`) pushes the prediction highly positive.
        3. **Stockpiling Indicator (Negative):** Conversely, if their rolling 30-day brand spend *share* (`brand_roll30_spend_share`) is extremely high, they are *less* likely to repeat. This implies they just stockpiled a massive haul of the brand and won't need to replenish soon.
        4. **Cannibalization Resistance:** Similar to acquisition, high `co_roll30_spend_share` (spending heavily on the parent company's *other* products) reduces the likelihood of repeating this specific brand.
        """)
    except Exception as e:
        callout(f"Could not load SHAP simulation data: {e}", "warning")
        
    st.markdown("---")
    st.subheader("4. Actionable Strategy: Optimizing a Fixed Budget")
    
    callout(
        "**Strategic Conclusion:** Because we cannot prove the coupon strictly *caused* the repeat purchase, we cannot use an Expected Value (ROI) optimization to calculate the exact profitability per coupon. Instead, we must treat the retention campaign as a **budget-constrained loyalty prioritization**.",
        kind="info"
    )
    
    st.markdown("""
    Given a fixed budget of $X (or N total coupons available):
    1. **Rank customers** by their predicted repeat probability.
    2. **Target the top N customers** to maximize the *observed volume* of transactions associated with the campaign, maximizing immediate cash-flow and campaign redemption metrics.
    3. **Future Verification:** To accurately measure the *incremental* value (and unlock pure mathematical profit optimization like we did for Acquisition), future campaigns MUST withhold coupons from a randomly selected 10% of prior buyers to establish a clean control baseline.
    """)
