import streamlit as st
from app.theme import headline

def render():
    headline("Confident acquisition targeting. Budget-prioritised retention targeting. Plus a recommended causal validation experiment for retention.")
    
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.subheader("Coupon Targeting Decision Matrix")
    
    st.markdown("""
    | Cohort | High predicted score (Top Deciles) | Low predicted score (Bottom Deciles) |
    |---|---|---|
    | **Acquisition** (never buyers) | ✅ **Send coupon**<br/>Causal ROI defensible. Represents true incremental acquisition. | ❌ **Suppress**<br/>Weak predicted causal effect. Save budget. |
    | **Retention** (prior buyers) | 🔵 **Send if budget allows**<br/>Highly likely to be observed as repeating. Operational prioritisation only. | 🔵 **De-prioritise**<br/>Less likely to be observed as repeating. Suppress if budget constrained. |
    """, unsafe_allow_html=True)
    
    st.markdown("<br/><br/>", unsafe_allow_html=True)
    st.subheader("Next Steps: Validating Retention Causal Impact")
    st.markdown("""
    To determine if retention coupons are actually driving *incremental* repeat trips (as opposed to subsidising natural behaviour), we must run a controlled experiment.
    
    **Experiment Design:**
    1. Define a specific post-offer observation window $W$ (e.g., 60 days).
    2. Randomly select 10% of eligible retention shoppers for a **Control Group**. Suppress their coupons.
    3. Serve the remaining 90% as the **Treatment Group** using the predictive model scores for ranking.
    4. At the end of window $W$, compare the repeat rates of the Control Group vs Treatment Group. The difference is the true incremental causal effect of the retention coupons.
    """)
