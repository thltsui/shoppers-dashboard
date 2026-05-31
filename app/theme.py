import streamlit as st

def setup_page(title: str):
    st.set_page_config(
        page_title=title,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def headline(text: str):
    st.markdown(f"<h3 style='margin-bottom: 2rem; font-weight: 600; line-height: 1.4;'>{text}</h3>", unsafe_allow_html=True)

def callout(text: str, kind: str = "info"):
    if kind == "warning":
        st.warning(text, icon="⚠️")
    elif kind == "success":
        st.success(text, icon="✅")
    else:
        st.info(text, icon="ℹ️")

def metric_tile(label: str, value: str):
    st.metric(label=label, value=value)
