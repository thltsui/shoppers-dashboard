import streamlit as st
from app.theme import setup_page
from app.panels import overview, cohort_split, acquisition, retention, policy

def main():
    setup_page("Coupon Strategy Dashboard")
    
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio(
        "Go to",
        [
            "1. Overview & Setup",
            "2. Cohort Split",
            "3. Acquisition (Causal)",
            "4. Retention (Descriptive)",
            "5. Unified Policy"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Methodology")
    st.sidebar.markdown(
        "This dashboard provides prescriptive, non-exploratory insights for coupon targeting. "
        "It applies fixed-effects within-offer ANCOVA to isolate shopper characteristics, and enforces "
        "a strict causal separation between never-buyers and prior-buyers."
    )
    
    if selection == "1. Overview & Setup":
        overview.render()
    elif selection == "2. Cohort Split":
        cohort_split.render()
    elif selection == "3. Acquisition (Causal)":
        acquisition.render()
    elif selection == "4. Retention (Descriptive)":
        retention.render()
    elif selection == "5. Unified Policy":
        policy.render()

if __name__ == "__main__":
    main()
