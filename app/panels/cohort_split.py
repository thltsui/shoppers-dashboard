import streamlit as st
import plotly.express as px
from app.data_loaders import load_train_sample
from app.theme import headline, callout

def render():
    headline("The Strategy: Answering the Core Business Questions")
    
    st.markdown("""
    To design a profitable coupon strategy, we must answer two fundamental questions:
    1. **Which coupons are actually effective at driving new behaviour?**
    2. **To whom should we give these coupons to maximize ROI?**
    
    To answer these accurately, we must split our shoppers into two distinct groups based on their past relationship with the offered brand. Treating them as a single group will lead to flawed business decisions.
    """)
    
    st.markdown("<hr/>", unsafe_allow_html=True)
    
    sample_df = load_train_sample()
    cohort_counts = sample_df["cohort"].value_counts().reset_index()
    
    fig = px.pie(
        cohort_counts, 
        values="count", 
        names="cohort", 
        title="Coupon Volume by Shopper History",
        color="cohort",
        color_discrete_map={"acquisition": "#2ca02c", "retention": "#ff7f0e"},
        hole=0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(
            "### 1. Acquisition (Never-Buyers)\n"
            "Shoppers who have **never** bought the offered brand before.\n\n"
            "---\n\n"
            "#### The Causal Assumption ✅\n"
            "We assume that without the coupon, a shopper's likelihood of spontaneously trying a brand they have never bought before is near zero (~2.4% baseline). Therefore, if they use the coupon and become a repeat buyer, we can confidently claim the **coupon caused the acquisition**.\n\n"
            "#### Business Value\n"
            "We can measure true incremental ROI and definitively answer which coupons are effective."
        )
        
    with col2:
        st.warning(
            "### 2. Retention (Prior-Buyers)\n"
            "Shoppers who have bought the offered brand in the past.\n\n"
            "---\n\n"
            "#### The Causal Limitation ❌\n"
            "Because these shoppers already buy the brand, we do not know their **counterfactual** (what they would have done without the coupon). Since everyone received a coupon (no control group), we cannot prove whether a repeat purchase was caused by the coupon, or if they were just restocking naturally.\n\n"
            "#### Business Value\n"
            "We cannot measure true ROI. We can only predict who will buy next to help prioritize fixed operational budgets."
        )
        
    st.markdown("<br/>", unsafe_allow_html=True)
    
    st.subheader("Why Retention is strictly descriptive: The Pasta Sauce Confounder")
    st.markdown("""
    When evaluating if a retention coupon "worked", the dataset simply tells us if the shopper repeated (`repeater = 1` or `0`) after receiving the offer. However, we are missing a critical piece of information: **the length of the observation window.**
    """)
    
    callout(
        "A shopper who buys bread every 4 days will almost certainly register as `repeater = 1` before the window closes. A shopper who buys pasta sauce every 40 days will likely register as `repeater = 0`. "
        "The model will learn that bread coupons are 'highly effective' and pasta sauce coupons are 'failures' — but in reality, it's just learning the natural category cycle.",
        kind="warning"
    )
