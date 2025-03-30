import streamlit as st
from ui import render_upload_page, render_clean_page, render_insights_page, render_predictive_page
from visualizations import render_visualization_page

# Set page configuration
st.set_page_config(page_title="AI Data Cleaner Pro", layout="wide", initial_sidebar_state="expanded")

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("Welcome to **AI Data Cleaner Pro**! Transform your data with AI magic.")
page = st.sidebar.radio("Go to", ["Upload", "Clean", "Insights", "Visualize", "Predictive", "Share"])

# Add feedback, community, and premium links
st.sidebar.markdown("---")
st.sidebar.markdown("**Feedback**")
st.sidebar.markdown("Help us improve! [Share your feedback](https://docs.google.com/forms/d/e/1FAIpQLScpUFM0Y5_i5LJDM-HZEZEtOHbLHy4Vp-ek_-819MRZo7Q9rQ/viewform?usp=dialog)")
st.sidebar.markdown("**Join Our Community**")
st.sidebar.markdown("Connect with others! [Join our Discord](https://discord.gg/your-invite-link)")
st.sidebar.markdown("**Upgrade to Premium**")
st.sidebar.markdown("Unlock advanced features for $5/month! [Upgrade Now](https://stripe.com/your-checkout-link)")
st.sidebar.markdown("**User Testimonials**")
st.sidebar.markdown("- 'This app replaced Excel for me!' - @DataNerd")
st.sidebar.markdown("- 'Mind-blowing AI features!' - @MLFan")

# Page routing
if page == "Upload":
    render_upload_page()
elif page == "Clean":
    render_clean_page()
elif page == "Insights":
    render_insights_page()
elif page == "Visualize":
    df = st.session_state.get('cleaned_df') if st.session_state.get('cleaned_df') is not None else st.session_state.get('df')
    render_visualization_page(df)
elif page == "Predictive":
    df = st.session_state.get('cleaned_df') if st.session_state.get('cleaned_df') is not None else st.session_state.get('df')
    render_predictive_page(df)
elif page == "Share":
    st.title("🌐 Share Your Work")
    st.write("Sharing and collaboration features coming soon! Stay tuned.")

# Custom CSS for styling
st.markdown("""
<style>
    .stCheckbox { margin-bottom: 10px; }
    .stRadio { margin-bottom: 10px; }
    .stSelectbox { margin-bottom: 10px; }
    .stMultiSelect { margin-bottom: 10px; }
    .stTextInput { margin-bottom: 10px; }
    .stButton { margin-top: 10px; }
    .welcome { font-size: 18px; color: #4CAF50; }
    div[data-baseweb="select"] > div {
        cursor: pointer !important;
    }
    div[data-baseweb="select"] > div:hover {
        background-color: #f0f0f0;
    }
    .main { background-color: #f9f9f9; padding: 20px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)