import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Data Toy", layout="wide", initial_sidebar_state="expanded")

import os
from typing import Optional
from ui import render_upload_page, render_clean_page, render_insights_page, render_predictive_page
from visualizations import render_visualization_page
from data_utils import chat_with_gpt, AI_AVAILABLE
import sqlite3
import pickle
import bcrypt
from authlib.integrations.requests_client import OAuth2Session
import requests

GOOGLE_CLIENT_ID = st.secrets["GOOGLE_CLIENT_ID"]
GOOGLE_CLIENT_SECRET = st.secrets["GOOGLE_CLIENT_SECRET"]
GOOGLE_REDIRECT_URI = "https://madhavvan-ai-data-cleaner-app-djmiue.streamlit.app"
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
SCOPES = ["openid", "email", "profile"]

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'theme' not in st.session_state:
    st.session_state.theme = "dark"
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'page' not in st.session_state:
    st.session_state.page = "Login"
if 'progress' not in st.session_state:
    st.session_state.progress = {
        "Upload": "Not Started",
        "Clean": "Not Started",
        "Insights": "Not Started",
        "Visualize": "Not Started",
        "Predictive": "Not Started",
        "Share": "Not Started"
    }
if 'user_info' not in st.session_state:
    st.session_state.user_info = None

def init_db():
    conn = sqlite3.connect('datatoy_users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, email TEXT, name TEXT, password TEXT, google_id TEXT, profile_picture TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS sessions 
                 (username TEXT PRIMARY KEY, session_data BLOB)''')
    conn.commit()
    conn.close()

init_db()

def add_user(username: str, email: str, name: str, password: str = None, google_id: str = None, profile_picture: str = None):
    hashed_password = None if password is None else bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    conn = sqlite3.connect('datatoy_users.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, email, name, password, google_id, profile_picture) VALUES (?, ?, ?, ?, ?, ?)", 
                  (username, email, name, hashed_password, google_id, profile_picture))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return False
    conn.close()
    return True

def verify_user(username: str, password: str) -> bool:
    conn = sqlite3.connect('datatoy_users.db')
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result and result[0]:
        stored_password = result[0]
        return bcrypt.checkpw(password.encode('utf-8'), stored_password)
    return False

def get_user_by_google_id(google_id: str):
    conn = sqlite3.connect('datatoy_users.db')
    c = conn.cursor()
    c.execute("SELECT username, email, name, profile_picture FROM users WHERE google_id = ?", (google_id,))
    result = c.fetchone()
    conn.close()
    return result

def save_session(username):
    session_data = {
        'df': st.session_state.get('df'),
        'cleaned_df': st.session_state.get('cleaned_df'),
        'logs': st.session_state.get('logs'),
        'suggestions': st.session_state.get('suggestions'),
        'previous_states': st.session_state.get('previous_states'),
        'redo_states': st.session_state.get('redo_states'),
        'chat_history': st.session_state.get('chat_history'),
        'cleaning_history': st.session_state.get('cleaning_history'),
        'cleaning_templates': st.session_state.get('cleaning_templates'),
        'is_premium': st.session_state.get('is_premium'),
        'ai_suggestions_used': st.session_state.get('ai_suggestions_used'),
        'dropped_columns': st.session_state.get('dropped_columns'),
        'progress': st.session_state.get('progress'),
        'dashboard_charts': st.session_state.get('dashboard_charts'),
        'dashboard_filters': st.session_state.get('dashboard_filters'),
        'authenticated': st.session_state.authenticated,
        'username': st.session_state.username,
        'user_info': st.session_state.user_info
    }
    session_blob = pickle.dumps(session_data)
    conn = sqlite3.connect('datatoy_users.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO sessions (username, session_data) VALUES (?, ?)", 
              (username, session_blob))
    conn.commit()
    conn.close()

def load_session(username):
    conn = sqlite3.connect('datatoy_users.db')
    c = conn.cursor()
    c.execute("SELECT session_data FROM sessions WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        session_data = pickle.loads(result[0])
        for key, value in session_data.items():
            st.session_state[key] = value

def load_css(theme: str = "dark") -> None:
    css = f"""
    body {{
        font-family: 'Roboto', sans-serif !important;
        margin: 0;
        padding: 0;
    }}
    body.{theme}-theme {{
        display: block !important;
    }}
    body.{theme}-theme .stApp {{
        background: linear-gradient(to bottom right, {'#1C2526, #2A3B47' if theme == 'dark' else '#F0F4F8, #D9E2EC'}) !important;
        color: {'#FFFFFF' if theme == 'dark' else '#000000'} !important;
    }}
    body.{theme}-theme .css-1d391kg {{
        background-color: {'#1C2526' if theme == 'dark' else '#D9E2EC'} !important;
        color: {'#FFFFFF' if theme == 'dark' else '#000000'} !important;
    }}
    body.{theme}-theme .css-1d391kg .tagline {{
        font-size: 16px !important;
        color: {'#1E90FF' if theme == 'dark' else '#0066CC'} !important;
        font-style: italic !important;
    }}
    body.{theme}-theme h1 {{
        color: {'#1E90FF' if theme == 'dark' else '#0066CC'} !important;
        font-family: 'Roboto', sans-serif !important;
    }}
    body.{theme}-theme h2, body.{theme}-theme h3 {{
        color: {'#FFD700' if theme == 'dark' else '#CC9900'} !important;
        font-family: 'Roboto', sans-serif !important;
    }}
    body.{theme}-theme .stButton > button {{
        background-color: {'#1E90FF' if theme == 'dark' else '#0066CC'} !important;
        color: white !important;
        border-radius: 5px !important;
        transition: background-color 0.3s !important;
        font-family: 'Roboto', sans-serif !important;
        border: none !important;
    }}
    body.{theme}-theme .stButton > button:hover {{
        background-color: {'#FFD700' if theme == 'dark' else '#CC9900'} !important;
        color: {'#1C2526' if theme == 'dark' else '#FFFFFF'} !important;
    }}
    body.{theme}-theme .stContainer {{
        background-color: {'rgba(255, 255, 255, 0.1)' if theme == 'dark' else 'rgba(0, 0, 0, 0.05)'} !important;
        border-radius: 10px !important;
        padding: 15px !important;
        box-shadow: 0 4px 6px {'rgba(0, 0, 0, 0.1)' if theme == 'dark' else 'rgba(0, 0, 0, 0.05)'} !important;
    }}
    body.{theme}-theme .stExpander {{
        background-color: {'rgba(255, 255, 255, 0.05)' if theme == 'dark' else 'rgba(0, 0, 0, 0.02)'} !important;
        border-radius: 10px !important;
    }}
    body.{theme}-theme .stTextInput > div > div > input,
    body.{theme}-theme .stSelectbox > div > div > div,
    body.{theme}-theme .stMultiSelect > div > div > div {{
        background-color: {'#2A3B47' if theme == 'dark' else '#F0F4F8'} !important;
        color: {'#FFFFFF' if theme == 'dark' else '#000000'} !important;
        border: 1px solid {'#1E90FF' if theme == 'dark' else '#0066CC'} !important;
        border-radius: 5px !important;
    }}
    body.{theme}-theme .stTextInput > div > div > input:focus,
    body.{theme}-theme .stSelectbox > div > div > div:focus,
    body.{theme}-theme .stMultiSelect > div > div > div:focus {{
        border-color: {'#FFD700' if theme == 'dark' else '#CC9900'} !important;
        outline: none !important;
        box-shadow: 0 0 5px {'rgba(255, 215, 0, 0.5)' if theme == 'dark' else 'rgba(204, 153, 0, 0.5)'} !important;
    }}
    body.{theme}-theme .stDataFrame {{
        background-color: {'#2A3B47' if theme == 'dark' else '#F0F4F8'} !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }}
    body.{theme}-theme .stCheckbox, body.{theme}-theme .stRadio {{
        margin-bottom: 10px !important;
    }}
    body.{theme}-theme .stProgress > div > div {{
        background-color: {'#1E90FF' if theme == 'dark' else '#0066CC'} !important;
    }}
    body.{theme}-theme .stAlert {{
        background-color: {'rgba(255, 255, 255, 0.1)' if theme == 'dark' else 'rgba(0, 0, 0, 0.05)'} !important;
        color: {'#FFFFFF' if theme == 'dark' else '#000000'} !important;
        border-radius: 5px !important;
    }}
    body.{theme}-theme div[data-baseweb="select"] > div {{
        cursor: pointer !important;
    }}
    body.{theme}-theme div[data-baseweb="select"] > div:hover {{
        background-color: {'#3C4F5C' if theme == 'dark' else '#D9E2EC'} !important;
    }}
    .google-login-button {{
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background-color: #FFFFFF !important;
        color: #757575 !important;
        border: 1px solid #DADCE0 !important;
        border-radius: 4px !important;
        padding: 10px 20px !important;
        font-size: 16px !important;
        font-family: 'Roboto', sans-serif !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        transition: background-color 0.3s ease, box-shadow 0.3s ease !important;
        width: 100% !important;
        box-sizing: border-box !important;
        margin: 10px auto !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1) !important;
    }}
    .google-login-button:hover {{
        background-color: #F8FAFC !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
    }}
    .google-login-button img {{
        width: 20px !important;
        height: 20px !important;
        margin-right: 10px !important;
    }}
    .google-login-button span {{
        color: #757575 !important;
        font-family: 'Roboto', sans-serif !important;
    }}
    a.google-login-button {{
        text-decoration: none !important;
    }}
    """
    components.html(
        f"""
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
        <style>
            {css}
        </style>
        <script>
            document.body.className = "{theme}-theme";
            console.log("Applied body class:", document.body.className);
            document.body.style.backgroundColor = "{ '#1C2526' if theme == 'dark' else '#F0F4F8' }";
        </script>
        """,
        height=0
    )

def render_custom_header(page_title: str) -> None:
    header = st.container()
    with header:
        st.markdown(f"<h1 style='margin-top: 20px;'>{page_title}</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid #FFD700;'>", unsafe_allow_html=True)

def get_google_auth_url():
    client = OAuth2Session(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, redirect_uri=GOOGLE_REDIRECT_URI, scope=SCOPES)
    auth_url, state = client.create_authorization_url(GOOGLE_AUTH_URL)
    st.session_state['oauth_state'] = state
    return auth_url

def handle_google_callback():
    client = OAuth2Session(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, redirect_uri=GOOGLE_REDIRECT_URI, state=st.session_state.get('oauth_state'))
    token = client.fetch_token(GOOGLE_TOKEN_URL, authorization_response=st.query_params['code'])
    user_info = requests.get(GOOGLE_USERINFO_URL, headers={'Authorization': f"Bearer {token['access_token']}"}).json()
    return user_info

if st.session_state.username:
    load_session(st.session_state.username)

if st.session_state.page == "Login":
    load_css(st.session_state.theme)
    st.markdown(
        f"""
        <div class="login-card" style="background: {'#2A3B47' if st.session_state.theme == 'dark' else '#FFFFFF'}; border-radius: 15px; padding: 30px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); max-width: 400px; margin: 0 auto; margin-top: 100px;">
        <h1 style="text-align: center; margin-bottom: 20px; font-size: 24px; color: {'#1E90FF' if st.session_state.theme == 'dark' else '#0066CC'}; font-family: 'Roboto', sans-serif;">Welcome to Data Toy AI</h1>
        """,
        unsafe_allow_html=True
    )
    
    username = st.text_input("Username", placeholder="Enter your username", key="username_input", help="Enter your username to log in.")
    password = st.text_input("Password", type="password", placeholder="Enter your password", key="password_input", help="Enter your password to log in.")
    
    if st.button("Login", key="login_button", help="Click to log in with your username and password."):
        if verify_user(username, password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.page = "Upload"
            st.session_state.user_info = None
            save_session(username)
            st.rerun()
        else:
            st.error("Incorrect username or password")

    auth_url = get_google_auth_url()
    st.markdown(
        f"""
        <a href="{auth_url}" target="_self" style="text-decoration: none;">
            <div class="google-login-button">
                <img src="https://developers.google.com/identity/images/g-logo.png" alt="Google Icon"/>
                <span>Sign in with Google</span>
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )

    if 'code' in st.query_params:
        user_info = handle_google_callback()
        google_id = user_info['sub']
        user = get_user_by_google_id(google_id)
        if user:
            username, email, name, profile_picture = user
        else:
            username = user_info['email'].split('@')[0]
            email = user_info['email']
            name = user_info['name']
            profile_picture = user_info.get('picture')
            add_user(username, email, name, google_id=google_id, profile_picture=profile_picture)
        st.session_state.authenticated = True
        st.session_state.username = username
        st.session_state.user_info = user_info
        st.session_state.page = "Upload"
        save_session(username)
        st.query_params.clear()
        st.rerun()

    if st.button("Sign Up", key="signup_button", help="Click to create a new account."):
        st.session_state.page = "Sign Up"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "Sign Up":
    load_css(st.session_state.theme)
    st.markdown(
        f"""
        <div class="login-card" style="background: {'#2A3B47' if st.session_state.theme == 'dark' else '#FFFFFF'}; border-radius: 15px; padding: 30px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); max-width: 400px; margin: 0 auto; margin-top: 100px;">
        <h1 style="text-align: center; margin-bottom: 20px; font-size: 24px; color: {'#1E90FF' if st.session_state.theme == 'dark' else '#0066CC'}; font-family: 'Roboto', sans-serif;">Sign Up for Data Toy AI</h1>
        """,
        unsafe_allow_html=True
    )
    
    new_username = st.text_input("New Username", placeholder="Choose a username", key="new_username_input", help="Choose a unique username for your account.")
    new_email = st.text_input("Email", placeholder="Enter your email", key="new_email_input", help="Enter your email address.")
    new_name = st.text_input("Name", placeholder="Enter your name", key="new_name_input", help="Enter your full name.")
    new_password = st.text_input("New Password", type="password", placeholder="Choose a password", key="new_password_input", help="Choose a secure password.")
    
    if st.button("Register", key="register_button", help="Click to register your account."):
        if add_user(new_username, new_email, new_name, new_password):
            st.success("Registration successful! Please log in.")
            st.session_state.page = "Login"
            st.rerun()
        else:
            st.error("Username already exists. Please choose a different username.")
    
    if st.button("Back to Login", key="back_to_login_button", help="Click to return to the login page."):
        st.session_state.page = "Login"
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.authenticated:
    load_css(st.session_state.theme)
    def setup_sidebar(logo_path: str = "images/datatoy_logo.png") -> Optional[str]:
        try:
            st.sidebar.image(logo_path, use_column_width=True)
        except FileNotFoundError:
            st.sidebar.markdown("**Data Toy** (Logo not found)", unsafe_allow_html=True)
            st.sidebar.warning(f"Logo file '{logo_path}' not found. Please add it to the project directory.")

        if st.session_state.user_info and 'picture' in st.session_state.user_info:
            st.sidebar.image(st.session_state.user_info['picture'], width=100, caption=f"Welcome, {st.session_state.user_info['name']}")
        else:
            st.sidebar.markdown(f"Welcome, {st.session_state.username}")

        st.sidebar.title("Navigation")
        st.sidebar.markdown("<p class='tagline'>Transform your data with AI magic.</p>", unsafe_allow_html=True)

        page = st.sidebar.radio("Go to", ["Upload", "Clean", "Insights", "Visualize", "Predictive", "Share"])

        st.sidebar.subheader("Theme")
        theme_choice = st.sidebar.selectbox("Select Theme", ["Dark", "Light"], index=0 if st.session_state.theme == "dark" else 1)
        if theme_choice == "Dark" and st.session_state.theme != "dark":
            st.session_state.theme = "dark"
            save_session(st.session_state.username)
            st.rerun()
        elif theme_choice == "Light" and st.session_state.theme != "light":
            st.session_state.theme = "light"
            save_session(st.session_state.username)
            st.rerun()

        st.sidebar.subheader("Your Progress")
        progress_text = ""
        for step, status in st.session_state.progress.items():
            emoji = "✅" if status == "Done" else "🟡" if status == "In Progress" else "⬜"
            progress_text += f"{emoji} {step}: {status}\n"
        st.sidebar.markdown(progress_text)

        if not AI_AVAILABLE:
            st.sidebar.error("⚠️ AI features are disabled. Please configure an OPENAI_API_KEY in .streamlit/secrets.toml or as an environment variable.")

        st.sidebar.subheader("AI Data Assistant")
        chat_container = st.sidebar.container()
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(f"**{message['role'].capitalize()}:** {message['content']}")

        chat_input = st.sidebar.chat_input("How can Data Toy help?")
        if chat_input:
            df = st.session_state.get('cleaned_df') if st.session_state.get('cleaned_df') is not None else st.session_state.get('df')
            if df is not None:
                st.session_state.chat_history.append({"role": "user", "content": chat_input})
                with st.spinner("Processing your query..."):
                    response = chat_with_gpt(df, chat_input)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                save_session(st.session_state.username)
                st.rerun()
            else:
                st.sidebar.warning("Please upload a dataset first to use the AI assistant.")

        st.sidebar.markdown("---")
        st.sidebar.markdown("**Feedback**")
        st.sidebar.markdown("Help us improve! [Share your feedback](https://docs.google.com/forms/d/e/1FAIpQLScpUFM0Y5_i5LJDM-HZEZEtOHbLHy4Vp-ek_-819MRZo7Q9rQ/viewform?usp=dialog)")
        st.sidebar.markdown("**Join Our Community**")
        st.sidebar.markdown("Connect with others! [Join our Discord](https://discord.gg/your-invite-link)")
        st.sidebar.markdown("**Upgrade to Premium**")
        st.sidebar.markdown("Unlock advanced features for $5/month! [Upgrade Now](https://stripe.com/your-checkout-link)")

        is_dev_mode = os.getenv("DEV_MODE") == "true"
        if is_dev_mode:
            st.sidebar.info("Running in DEV_MODE: Unlimited AI suggestions enabled.")

        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.user_info = None
            st.session_state.page = "Login"
            save_session(st.session_state.username if st.session_state.username else "default_user")
            st.rerun()

        return page

    def main() -> None:
        page = setup_sidebar()
        if not page:
            st.error("No page selected. Please select a page from the sidebar.")
            return

        page_titles = {
            "Upload": "Upload Your Dataset",
            "Clean": "Clean Your Dataset",
            "Insights": "Insights Dashboard",
            "Visualize": "Visualize Your Dataset",
            "Predictive": "Predictive Analytics",
            "Share": "Share Your Work"
        }
        render_custom_header(page_titles.get(page, "Data Toy"))

        try:
            if page == "Upload":
                render_upload_page()
            elif page == "Clean":
                render_clean_page()
            elif page == "Insights":
                render_insights_page()
            elif page == "Visualize":
                df = st.session_state.get('cleaned_df') if st.session_state.get('cleaned_df') is not None else st.session_state.get('df')
                if df is None:
                    st.warning("Please upload a dataset first on the Upload page.")
                else:
                    render_visualization_page(df)
            elif page == "Predictive":
                df = st.session_state.get('cleaned_df') if st.session_state.get('cleaned_df') is not None else st.session_state.get('df')
                if df is None:
                    st.warning("Please upload a dataset first on the Upload page.")
                else:
                    render_predictive_page(df)
            elif page == "Share":
                st.write("Sharing and collaboration features coming soon! Stay tuned.")
                st.session_state.progress["Share"] = "Done"
        except Exception as e:
            st.error(f"An error occurred while rendering the {page} page: {str(e)}. Please try again or contact support.")
            st.session_state.progress[page] = "Failed"

        save_session(st.session_state.username)

    if __name__ == "__main__":
        main()