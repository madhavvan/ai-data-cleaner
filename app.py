import streamlit as st
from ui import render_upload_page, render_clean_page
from visualizations import render_visualization_page

# Removed 'theme' argument since it's not supported
st.set_page_config(page_title="AI Data Cleaner", layout="wide", initial_sidebar_state="expanded")

st.sidebar.title("Navigation")
st.sidebar.markdown("Welcome to **AI Data Cleaner**! Transform your data with AI magic.")
page = st.sidebar.radio("Go to", ["Upload", "Clean", "Visualize"])

# Theme toggle removed since Streamlit doesn't support it natively in this way
# Keeping the button for future potential, but it won't do anything yet
if st.sidebar.button("Toggle Dark Mode"):
    st.session_state.theme = "dark" if st.session_state.get("theme", "light") == "light" else "light"
    st.sidebar.info("Theme switching not fully supported yet—stay tuned!")

if page == "Upload":
    render_upload_page()
elif page == "Clean":
    render_clean_page()
elif page == "Visualize":
    df = st.session_state.get('cleaned_df') if st.session_state.get('cleaned_df') is not None else st.session_state.get('df')
    render_visualization_page(df)

st.markdown("""
<style>
    .stCheckbox { margin-bottom: 10px; }
    .stRadio { margin-bottom: 10px; }
    .stSelectbox { margin-bottom: 10px; }
    .stTextInput { margin-bottom: 10px; }
    .stButton { margin-top: 10px; }
    .welcome { font-size: 18px; color: #4CAF50; }
</style>
""", unsafe_allow_html=True)