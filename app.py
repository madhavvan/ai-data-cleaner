import streamlit as st
from ui import render_upload_page, render_clean_page
from visualizations import render_visualization_page

# Set page configuration
st.set_page_config(page_title="AI Data Cleaner", layout="wide", initial_sidebar_state="expanded")

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("Welcome to **AI Data Cleaner**! Transform your data with AI magic.")
page = st.sidebar.radio("Go to", ["Upload", "Clean", "Visualize"])

# Add feedback form link (replace with your actual Google Form link)
st.sidebar.markdown("---")  # Separator
st.sidebar.markdown("**Feedback**")
st.sidebar.markdown("Help us improve! [Share your feedback](https://docs.google.com/forms/d/e/1FAIpQLScpUFM0Y5_i5LJDM-HZEZEtOHbLHy4Vp-ek_-819MRZo7Q9rQ/viewform?usp=dialog)")  # Replace with your actual Google Form link

# Theme toggle (kept as a placeholder)
if st.sidebar.button("Toggle Dark Mode"):
    st.session_state.theme = "dark" if st.session_state.get("theme", "light") == "light" else "light"
    st.sidebar.info("Theme switching not fully supported yet—stay tuned!")

# Page routing
if page == "Upload":
    render_upload_page()
elif page == "Clean":
    render_clean_page()
elif page == "Visualize":
    df = st.session_state.get('cleaned_df') if st.session_state.get('cleaned_df') is not None else st.session_state.get('df')
    render_visualization_page(df)

# Custom CSS for styling (including dropdown cursor fix)
st.markdown("""
<style>
    .stCheckbox { margin-bottom: 10px; }
    .stRadio { margin-bottom: 10px; }
    .stSelectbox { margin-bottom: 10px; }
    .stMultiSelect { margin-bottom: 10px; }
    .stTextInput { margin-bottom: 10px; }
    .stButton { margin-top: 10px; }
    .welcome { font-size: 18px; color: #4CAF50; }
    /* Fix dropdown cursor */
    div[data-baseweb="select"] > div {
        cursor: pointer !important;
    }
    div[data-baseweb="select"] > div:hover {
        background-color: #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)