import streamlit as st
import plotly.express as px
from app.data_loaders import load_anova, load_train_sample
from app.theme import metric_tile, callout, headline

def render():
    headline("Context & Setup: Understanding the Coupon Dataset")

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_tile("Transactions", "350M+")
    with col2:
        metric_tile("Shoppers", "160K")
    with col3:
        metric_tile("Distinct Offers", "37")

    sample_df = load_train_sample()
    sample_df["offer"] = sample_df["offer"].astype(str)

    st.markdown("---")
    st.subheader("1. The Setup: Offers, Objectives, and Timeline")
    st.markdown("""
    **Timeline & Data Range:** We utilize approximately 12 months of historical grocery transaction data to profile the shoppers. The 37 distinct coupon offers were distributed to these shoppers during a 60-day window between **March 1, 2013 and April 30, 2013**.
    
    **The Target Variables (What are we predicting?):** 
    After a shopper receives a coupon, their subsequent purchases are tracked to see if they bought the brand. We are provided two outcome metrics:
    *   **`repeater`**: A binary flag (Yes/No) indicating whether the shopper made at least one purchase of the offered brand.
    *   **`repeattrips`**: An integer count of exactly *how many* trips they made to buy the offered brand.
    
    ⚠️ **Critical Data Limitation:** The dataset does *not* define the length of the observation window used to track these outcomes (e.g., whether they tracked shoppers for 30 days or 90 days post-offer). This missing information severely limits our ability to draw certain causal conclusions, as we will explore in the next section.
    
    *(Note: This strategy dashboard focuses entirely on predicting the binary `repeater` outcome.)*
    
    **The Offers:** 
    A coupon is a discount targeted to a **specific customer** for a **specific product** (defined by its category, company, and brand). Let's look at how heavily each of the 37 offers was distributed, grouped by the product category.
    """)
    
    # Offer breakdown chart
    offer_counts = sample_df.groupby(["offer", "offered_category"]).size().reset_index(name="count")
    
    fig_offers = px.pie(
        offer_counts, 
        values="count", 
        names="offer",
        color="offered_category",
        title="Distribution of the 37 Offers (Colored by Product Category)",
        hole=0.4,
        hover_data=["offered_category"],
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig_offers.update_traces(textposition='inside', textinfo='percent+label')
    fig_offers.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_offers, use_container_width=True)

    st.markdown("---")
    st.subheader("2. The Shoppers (Customer Profiling)")
    st.markdown("""
    Before a shopper receives an offer, we extract their entire purchasing history up to that exact day. 
    This allows us to build a rich profile of their baseline behaviour. Click the categories below to explore the types of features we generated:
    """)
    
    with st.expander("💸 Spending Features (Wallet Size & Value)", expanded=True):
        st.markdown("""
        These features capture the monetary value and typical basket sizes of the shopper, calculated across their lifetime and recent 30-day history.
        
        *   **Store-wide Spend:** `total_prior_spend`, `total_prior_spend_30d`
        *   **Entity Spend:** `cat_prior_spend`, `cat_prior_spend_30d`, `co_prior_spend`, `co_prior_spend_30d`, `brand_prior_spend`, `brand_prior_spend_30d`
        *   **Average Basket Spend:** `cat_avg_spend_per_shop`, `cat_avg_spend_per_shop_30d`, `co_avg_spend_per_shop`, `co_avg_spend_per_shop_30d`, `brand_avg_spend_per_shop`, `brand_avg_spend_per_shop_30d`
        *   **Customer Segmentation:** `customer_premium_tier`
        """)
        
    with st.expander("❤️ Loyalty & Engagement Features"):
        st.markdown("""
        These features capture visit frequency, total items bought, and how much of their wallet is dedicated to specific brands and categories.
        
        *   **Store-wide Engagement:** `total_prior_quantity`, `total_prior_quantity_30d`, `total_prior_transactions`, `total_prior_transactions_30d`, `total_prior_visits`, `total_prior_visits_30d`
        *   **Entity Visits:** `cat_prior_visits`, `cat_prior_visits_30d`, `co_prior_visits`, `co_prior_visits_30d`, `brand_prior_visits`, `brand_prior_visits_30d`
        *   **Entity Quantity:** `cat_prior_qty`, `cat_prior_qty_30d`, `co_prior_qty`, `co_prior_qty_30d`, `brand_prior_qty`, `brand_prior_qty_30d`
        *   **Wallet Share:** `cat_wallet_share`, `cat_wallet_share_30d`, `co_wallet_share`, `co_wallet_share_30d`, `brand_wallet_share`, `brand_wallet_share_30d`
        *   **Momentum:** `cat_wallet_share_momentum`
        *   **Affinity Sizes:** `cat_size_per_affinity_visit`, `co_size_per_affinity_visit`, `brand_size_per_affinity_visit` (and 30d variants)
        """)
        
    with st.expander("🛒 Behavioural, Pricing & Cycle Features"):
        st.markdown("""
        These features capture the shopper's natural purchase cadence, price sensitivity, and brand promiscuity.
        
        *   **Purchase Cycles:** `days_since_first_purchase`, `recency`, `cat_recency`, `cat_avg_cycle_days`, `offer_delay_vs_cycle`
        *   **Brand Promiscuity:** `cat_unique_brands`, `cat_brand_variety`
        *   **Pricing & Promos:** `cat_roll30_unit_price`, `co_roll30_unit_price`, `brand_roll30_unit_price`
        *   **Market Share Context:** `cat_roll30_spend_share`, `cat_roll30_tx_share`, `co_roll30_spend_share`, `co_roll30_tx_share`, `brand_roll30_spend_share`, `brand_roll30_tx_share`
        *   **Discount Sensitivity:** `brand_premium_index`, `relative_discount_depth`
        """)

    st.markdown("---")
    st.subheader("3. The Catch: Targeted Assignment (No Control Group)")
    
    callout(
        "**Crucial Insight**: Coupons were NOT distributed randomly. They were targeted at specific audiences based on their profiles. "
        "Because of this, we cannot simply compare the success rate of Offer A vs Offer B, because the people who received them are completely different.",
        kind="warning"
    )

    st.markdown("To see this visually, let's look at the actual distributions for two of the most frequent offers in the dataset: **Offer 1197502** and **Offer 1208329**.")

    offers_to_compare = ["1197502", "1208329"]
    df_compare = sample_df[sample_df["offer"].isin(offers_to_compare)].copy()
    
    # Clip extreme outliers for better visualization
    df_compare["total_prior_spend_clip"] = df_compare["total_prior_spend"].clip(upper=df_compare["total_prior_spend"].quantile(0.95))
    df_compare["cat_prior_visits_clip"] = df_compare["cat_prior_visits"].clip(upper=df_compare["cat_prior_visits"].quantile(0.95))

    colA, colB = st.columns(2)
    
    with colA:
        fig_spend = px.histogram(
            df_compare, 
            x="total_prior_spend_clip", 
            color="offer", 
            barmode="overlay", 
            histnorm="probability density",
            title="Total Prior Spend ($)",
            labels={"total_prior_spend_clip": "Spend ($)"}
        )
        fig_spend.update_layout(margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_spend, use_container_width=True)
        
    with colB:
        fig_visits = px.histogram(
            df_compare, 
            x="cat_prior_visits_clip", 
            color="offer", 
            barmode="overlay", 
            histnorm="probability density",
            title="Category Prior Visits",
            labels={"cat_prior_visits_clip": "Visits"}
        )
        fig_visits.update_layout(margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_visits, use_container_width=True)

    st.markdown("""
    As the density plots show, these two offers were distributed to shoppers with entirely different baselines. 
    If one offer has a higher repeat rate, is it because the coupon was better, or because the shoppers were already heavier buyers?
    
    *Statistical check: An ANOVA test confirms that shopper profiles differ significantly (p < 0.001) across **every single observed characteristic** based on which offer they received.*
    """)
    
    with st.expander("View Statistical Proof (ANOVA F-Statistics)"):
        anova = load_anova()
        fig_anova = px.bar(
            anova.sort_values("F-statistic", ascending=True),
            x="F-statistic", y="Feature",
            orientation="h",
            text=anova["p-value"].apply(lambda p: f"p < 0.001" if p < 0.001 else f"p = {p:.3f}"),
            labels={"F-statistic": "ANOVA F-statistic (Higher = More Variance Across Offers)", "Feature": ""},
            color_discrete_sequence=["#4c78a8"]
        )
        fig_anova.update_traces(textposition="outside")
        fig_anova.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_anova, use_container_width=True)
