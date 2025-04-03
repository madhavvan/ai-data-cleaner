import streamlit as st
import os
from ui import render_upload_page, render_clean_page, render_insights_page, render_predictive_page
from visualizations import render_visualization_page
from data_utils import chat_with_gpt, AI_AVAILABLE  # Added AI_AVAILABLE import

# Set page configuration
st.set_page_config(page_title="Datatoy", layout="wide", initial_sidebar_state="expanded")

# Function to render a custom header with the Datatoy logo
def render_custom_header(page_title):
    # Create a container for the header with a gradient background
    header = st.container()
    with header:
        # Use columns to layout the logo and title
        col1, col2 = st.columns([1, 4])
        with col1:
            try:
                st.image("images/datatoy_logo.png", width=100)
            except FileNotFoundError:
                st.markdown("**Datatoy** (Logo not found)", unsafe_allow_html=True)
                st.warning("Logo file 'datatoy_logo.png' not found. Please add it to the project directory.")
        with col2:
            st.markdown(f"<h1 style='color: #1E90FF; margin-top: 20px;'>{page_title}</h1>", unsafe_allow_html=True)
    # Add a horizontal line below the header for separation
    st.markdown("<hr style='border: 1px solid #FFD700;'>", unsafe_allow_html=True)

# Add the Datatoy logo with error handling
try:
    st.sidebar.image("images/datatoy_logo.png", use_column_width=True)
except FileNotFoundError:
    st.sidebar.markdown("**Datatoy** (Logo not found)", unsafe_allow_html=True)
    st.sidebar.warning("Logo file 'datatoy_logo.png' not found. Please add it to the project directory.")

# Sidebar navigation
st.sidebar.title("Navigation")

# Add a styled tagline below the logo
st.sidebar.markdown("<p class='tagline'>Transform your data with AI magic.</p>", unsafe_allow_html=True)

page = st.sidebar.radio("Go to", ["Upload", "Clean", "Insights", "Visualize", "Predictive", "Share"])

# Enhancement: Warn if AI features are disabled
if not AI_AVAILABLE:
    st.sidebar.warning("AI features are disabled. Please configure an OPENAI_API_KEY in .streamlit/secrets.toml or as an environment variable.")

# AI Assistant in Sidebar
st.sidebar.subheader("AI Data Assistant")
chat_container = st.sidebar.container()
with chat_container:
    for message in st.session_state.get('chat_history', []):
        with st.chat_message(message["role"]):
            st.write(f"**{message['role'].capitalize()}:** {message['content']}")

chat_input = st.sidebar.chat_input("Ask me anything about your data!")
if chat_input:
    df = st.session_state.get('cleaned_df') if st.session_state.get('cleaned_df') is not None else st.session_state.get('df')
    if df is not None:
        st.session_state.chat_history.append({"role": "user", "content": chat_input})
        response = chat_with_gpt(df, chat_input)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
    else:
        st.sidebar.warning("Please upload a dataset first.")

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

# Debug mode check
is_dev_mode = os.getenv("DEV_MODE") == "true"
if is_dev_mode:
    st.sidebar.info("Running in DEV_MODE: Unlimited AI suggestions enabled.")

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
    # Custom header for the Share page
    render_custom_header("🌐 Share Your Work")
    st.write("Sharing and collaboration features coming soon! Stay tuned.")

# Custom CSS for styling
st.markdown("""
<style>
    /* Global app styling */
    .stApp {
        background: linear-gradient(to bottom right, #1C2526, #2A3B47); /* Gradient background inspired by the logo */
        color: #FFFFFF; /* White text for contrast */
    }
    /* Sidebar styling */
    .css-1d391kg { /* Sidebar class in Streamlit */
        background-color: #1C2526; /* Dark blue background */
        color: #FFFFFF;
    }
    .css-1d391kg .tagline {
        font-size: 16px;
        color: #1E90FF; /* Blue color matching the 'data' part of the logo */
        font-style: italic;
    }
    /* Header styling */
    h1 {
        color: #1E90FF; /* Blue color for page titles */
        font-family: 'Roboto', sans-serif;
    }
    /* Subheader styling */
    h2, h3 {
        color: #FFD700; /* Gold color matching the 'y' in the logo */
        font-family: 'Roboto', sans-serif;
    }
    /* Button styling */
    .stButton > button {
        background-color: #1E90FF; /* Blue background */
        color: white;
        border-radius: 5px;
        transition: background-color 0.3s;
        font-family: 'Roboto', sans-serif;
    }
    .stButton > button:hover {
        background-color: #FFD700; /* Gold on hover */
        color: #1C2526;
    }
    /* Container styling */
    .stContainer {
        background-color: rgba(255, 255, 255, 0.1); /* Semi-transparent white */
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    /* Expander styling */
    .stExpander {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    /* Text input and select box styling */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stMultiSelect > div > div > div {
        background-color: #2A3B47; /* Dark background */
        color: #FFFFFF;
        border: 1px solid #1E90FF; /* Blue border */
        border-radius: 5px;
    }
    /* Dataframe styling */
    .stDataFrame {
        background-color: #2A3B47;
        border-radius: 10px;
        padding: 10px;
    }
    /* Checkbox and radio styling */
    .stCheckbox, .stRadio {
        margin-bottom: 10px;
    }
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: #1E90FF;
    }
    /* Warning and info message styling */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.1);
        color: #FFFFFF;
        border-radius: 5px;
    }
    /* Hover effects for select boxes */
    div[data-baseweb="select"] > div {
        cursor: pointer !important;
    }
    div[data-baseweb="select"] > div:hover {
        background-color: #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)