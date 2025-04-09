import streamlit as st
import streamlit.components.v1 as components
import psycopg2
from psycopg2 import sql
import uuid  # For generating session tokens
import logging
from logging.handlers import RotatingFileHandler
import os
from typing import Optional
from ui import render_upload_page, render_clean_page, render_insights_page, render_predictive_page
from visualizations import render_visualization_page
from data_utils import chat_with_gpt, AI_AVAILABLE
import pickle
import bcrypt
from authlib.integrations.requests_client import OAuth2Session
import requests

# Set up logging with rotation
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = RotatingFileHandler('app.log', maxBytes=5*1024*1024, backupCount=3)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# Set page configuration as the first command
st.set_page_config(page_title="Data Toy", layout="wide", initial_sidebar_state="expanded")

# Google OAuth Configuration
GOOGLE_CLIENT_ID = st.secrets["GOOGLE_CLIENT_ID"]
GOOGLE_CLIENT_SECRET = st.secrets["GOOGLE_CLIENT_SECRET"]
GOOGLE_REDIRECT_URI = "https://madhavvan-ai-data-cleaner-app-djmiue.streamlit.app"
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
SCOPES = ["openid", "email", "profile"]

# Initialize session state at the top
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
if 'session_token' not in st.session_state:
    st.session_state.session_token = None
# Fix 6: Add flag to prevent OAuth callback race condition
if 'oauth_processed' not in st.session_state:
    st.session_state.oauth_processed = False

# Database connection using st.secrets with improved error handling
def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname=st.secrets["DB_NAME"],
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASSWORD"],
            host=st.secrets["DB_HOST"],
            port=st.secrets["DB_PORT"],
            sslmode="require"
        )
        return conn
    except psycopg2.Error as e:
        # Fix 3: Enhanced error handling with specific user feedback
        st.error(f"Database connection failed: {str(e)}. Please check your credentials or network.")
        logger.error(f"Database connection failed: {str(e)}")
        return None

def init_db():
    conn = get_db_connection()
    if conn is None:
        st.error("Failed to initialize database due to connection issues.")
        logger.error("Failed to initialize database due to connection failure")
        return
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, email TEXT, name TEXT, password BYTEA, google_id TEXT, profile_picture TEXT)''')
    c.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'sessions')")
    table_exists = c.fetchone()[0]
    if table_exists:
        c.execute("SELECT EXISTS (SELECT FROM information_schema.columns WHERE table_name = 'sessions' AND column_name = 'session_token')")
        session_token_exists = c.fetchone()[0]
        if not session_token_exists:
            c.execute("ALTER TABLE sessions ADD COLUMN session_token TEXT")
            logger.debug("Added session_token column to sessions table")
    else:
        c.execute('''CREATE TABLE sessions 
                     (username TEXT PRIMARY KEY, session_token TEXT, session_data BYTEA)''')
        logger.debug("Created sessions table with session_token column")
    conn.commit()
    conn.close()

init_db()

# Fix 2: Add compatibility for st.query_params across Streamlit versions
def get_query_param(key, default=None):
    try:
        # Fix 1: Correct st.query_params usage
        return st.query_params.get(key, default)
    except AttributeError:
        # Fallback for older Streamlit versions
        return st.experimental_get_query_params().get(key, [default])[0]

def restore_session():
    logger.debug("Starting restore_session")
    # Fix 1 & 2: Use corrected query params with compatibility
    session_token = get_query_param('session_token')
    logger.debug(f"Session token from query params: {session_token}")
    if session_token:
        conn = get_db_connection()
        if conn is None:
            logger.debug("Failed to connect to database in restore_session")
            return
        c = conn.cursor()
        try:
            c.execute("SELECT username, session_data FROM sessions WHERE session_token = %s", (session_token,))
            result = c.fetchone()
            logger.debug(f"Database query result: {result}")
            if result:
                username, session_data = result
                session_data = pickle.loads(session_data)
                st.session_state.authenticated = session_data.get('authenticated', False)
                st.session_state.username = username
                st.session_state.user_info = session_data.get('user_info', None)
                st.session_state.session_token = session_token
                st.session_state.page = session_data.get('page', "Upload")
                # Fix 7: Load DataFrame paths instead of full objects
                df_path = session_data.get('df_path')
                cleaned_df_path = session_data.get('cleaned_df_path')
                if df_path and os.path.exists(df_path):
                    st.session_state.df = pd.read_pickle(df_path)
                if cleaned_df_path and os.path.exists(cleaned_df_path):
                    st.session_state.cleaned_df = pd.read_pickle(cleaned_df_path)
                for key, value in session_data.items():
                    if key not in ['authenticated', 'username', 'user_info', 'session_token', 'page', 'df', 'cleaned_df', 'df_path', 'cleaned_df_path']:
                        st.session_state[key] = value
                logger.info(f"Session restored for user {username}")
            else:
                logger.debug("No session found for the given session token")
        except Exception as e:
            logger.error(f"Error in restore_session: {str(e)}")
        finally:
            conn.close()

def save_auth_state():
    if st.session_state.username:
        logger.debug("Starting save_auth_state")
        conn = get_db_connection()
        if conn is None:
            logger.debug("Failed to connect to database in save_auth_state")
            return
        c = conn.cursor()
        try:
            # Fix 4: Check for existing session token before generating a new one
            c.execute("SELECT session_token FROM sessions WHERE username = %s", (st.session_state.username,))
            existing_token = c.fetchone()
            if existing_token and existing_token[0]:
                st.session_state.session_token = existing_token[0]
            elif not st.session_state.session_token:
                st.session_state.session_token = str(uuid.uuid4())
                logger.debug(f"Generated new session token: {st.session_state.session_token}")
            # Fix 1 & 2: Use corrected query params
            st.query_params['session_token'] = st.session_state.session_token

            # Fix 7: Offload DataFrames to disk
            session_data = {k: v for k, v in st.session_state.items() if k not in ['df', 'cleaned_df']}
            df_path = f"temp_df_{st.session_state.username}.pkl"
            cleaned_df_path = f"temp_cleaned_df_{st.session_state.username}.pkl"
            if 'df' in st.session_state and st.session_state.df is not None:
                st.session_state.df.to_pickle(df_path)
                session_data['df_path'] = df_path
            if 'cleaned_df' in st.session_state and st.session_state.cleaned_df is not None:
                st.session_state.cleaned_df.to_pickle(cleaned_df_path)
                session_data['cleaned_df_path'] = cleaned_df_path
            session_blob = pickle.dumps(session_data)
            c.execute("INSERT INTO sessions (username, session_token, session_data) VALUES (%s, %s, %s) ON CONFLICT (username) DO UPDATE SET session_token = %s, session_data = %s",
                      (st.session_state.username, st.session_state.session_token, session_blob, st.session_state.session_token, session_blob))
            conn.commit()
            logger.info("Session state saved successfully")
        except Exception as e:
            logger.error(f"Error in save_auth_state: {str(e)}")
        finally:
            conn.close()

restore_session()

def add_user(username: str, email: str, name: str, password: str = None, google_id: str = None, profile_picture: str = None):
    hashed_password = None if password is None else bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    conn = get_db_connection()
    if conn is None:
        return False
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (username, email, name, password, google_id, profile_picture) VALUES (%s, %s, %s, %s, %s, %s)",
            (username, email, name, hashed_password if hashed_password is None else psycopg2.Binary(hashed_password), google_id, profile_picture)
        )
        conn.commit()
    except psycopg2.IntegrityError:
        conn.close()
        return False
    conn.close()
    return True

def verify_user(username: str, password: str) -> bool:
    conn = get_db_connection()
    if conn is None:
        return False
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = %s", (username,))
    result = c.fetchone()
    conn.close()
    if result and result[0]:
        stored_password = result[0]
        if stored_password is None:
            return False
        if isinstance(stored_password, memoryview):
            stored_password = stored_password.tobytes()
        if isinstance(stored_password, str):
            stored_password = stored_password.encode('utf-8')
        return bcrypt.checkpw(password.encode('utf-8'), stored_password)
    return False

def get_user_by_google_id(google_id: str):
    conn = get_db_connection()
    if conn is None:
        return None
    c = conn.cursor()
    c.execute("SELECT username, email, name, profile_picture FROM users WHERE google_id = %s", (google_id,))
    result = c.fetchone()
    conn.close()
    return result

def save_session(username):
    save_auth_state()

def load_session(username):
    conn = get_db_connection()
    if conn is None:
        return
    c = conn.cursor()
    c.execute("SELECT session_data FROM sessions WHERE username = %s", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        session_data = pickle.loads(result[0])
        for key, value in session_data.items():
            if key not in ['authenticated', 'username', 'user_info', 'session_token', 'page']:
                st.session_state[key] = value

def load_css(theme: str = "dark") -> None:
    css = """
    body {
        font-family: 'Roboto', sans-serif !important;
        margin: 0;
        padding: 0;
    }

    body.{theme}-theme {
        display: block !important;
    }

    /* App-wide styles for dark theme */
    body.dark-theme .stApp {
        background: linear-gradient(to bottom right, #1C2526, #2A3B47) !important;
        color: #FFFFFF !important;
    }

    body.dark-theme .css-1d391kg {
        background-color: #1C2526 !important;
        color: #FFFFFF !important;
    }

    body.dark-theme .css-1d391kg .tagline {
        font-size: 16px !important;
        color: #1E90FF !important;
        font-style: italic !important;
    }

    body.dark-theme h1 {
        color: #1E90FF !important;
        font-family: 'Roboto', sans-serif !important;
    }

    body.dark-theme h2, body.dark-theme h3 {
        color: #FFD700 !important;
        font-family: 'Roboto', sans-serif !important;
    }

    body.dark-theme .stButton > button {
        background-color: #1E90FF !important;
        color: white !important;
        border-radius: 5px !important;
        transition: background-color 0.3s !important;
        font-family: 'Roboto', sans-serif !important;
        border: none !important;
    }

    body.dark-theme .stButton > button:hover {
        background-color: #FFD700 !important;
        color: #1C2526 !important;
    }

    body.dark-theme .stContainer {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        padding: 15px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }

    body.dark-theme .stExpander {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px !important;
    }

    body.dark-theme .stTextInput > div > div > input,
    body.dark-theme .stSelectbox > div > div > div,
    body.dark-theme .stMultiSelect > div > div > div {
        background-color: #2A3B47 !important;
        color: #FFFFFF !important;
        border: 1px solid #1E90FF !important;
        border-radius: 5px !important;
    }

    body.dark-theme .stDataFrame {
        background-color: #2A3B47 !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }

    body.dark-theme .stCheckbox, body.dark-theme .stRadio {
        margin-bottom: 10px !important;
    }

    body.dark-theme .stProgress > div > div {
        background-color: #1E90FF !important;
    }

    body.dark-theme .stAlert {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #FFFFFF !important;
        border-radius: 5px !important;
    }

    body.dark-theme div[data-baseweb="select"] > div {
        cursor: pointer !important;
    }

    body.dark-theme div[data-baseweb="select"] > div:hover {
        background-color: #3C4F5C !important;
    }

    /* App-wide styles for light theme */
    body.light-theme .stApp {
        background: linear-gradient(to bottom right, #F0F4F8, #D9E2EC) !important;
        color: #000000 !important;
    }

    body.light-theme .css-1d391kg {
        background-color: #D9E2EC !important;
        color: #000000 !important;
    }

    body.light-theme .css-1d391kg .tagline {
        font-size: 16px !important;
        color: #0066CC !important;
        font-style: italic !important;
    }

    body.light-theme h1 {
        color: #0066CC !important;
        font-family: 'Roboto', sans-serif !important;
    }

    body.light-theme h2, body.light-theme h3 {
        color: #CC9900 !important;
        font-family: 'Roboto', sans-serif !important;
    }

    body.light-theme .stButton > button {
        background-color: #0066CC !important;
        color: white !important;
        border-radius: 5px !important;
        transition: background-color 0.3s !important;
        font-family: 'Roboto', sans-serif !important;
        border: none !important;
    }

    body.light-theme .stButton > button:hover {
        background-color: #CC9900 !important;
        color: #FFFFFF !important;
    }

    body.light-theme .stContainer {
        background-color: rgba(0, 0, 0, 0.05) !important;
        border-radius: 10px !important;
        padding: 15px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important;
    }

    body.light-theme .stExpander {
        background-color: rgba(0, 0, 0, 0.02) !important;
        border-radius: 10px !important;
    }

    body.light-theme .stTextInput > div > div > input,
    body.light-theme .stSelectbox > div > div > div,
    body.light-theme .stMultiSelect > div > div > div {
        background-color: #F0F4F8 !important;
        color: #000000 !important;
        border: 1px solid #0066CC !important;
        border-radius: 5px !important;
    }

    body.light-theme .stDataFrame {
        background-color: #F0F4F8 !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }

    body.light-theme .stCheckbox, body.light-theme .stRadio {
        margin-bottom: 10px !important;
    }

    body.light-theme .stProgress > div > div {
        background-color: #0066CC !important;
    }

    body.light-theme .stAlert {
        background-color: rgba(0, 0, 0, 0.05) !important;
        color: #000000 !important;
        border-radius: 5px !important;
    }

    body.light-theme div[data-baseweb="select"] > div {
        cursor: pointer !important;
    }

    body.light-theme div[data-baseweb="select"] > div:hover {
        background-color: #D9E2EC !important;
    }

    /* Google Login Button Styling (same for both themes) */
    .google-login-button {
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
    }

    .google-login-button:hover {
        background-color: #F8FAFC !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
    }

    .google-login-button img {
        width: 20px !important;
        height: 20px !important;
        margin-right: 10px !important;
    }

    .google-login-button span {
        color: #757575 !important;
        font-family: 'Roboto', sans-serif !important;
    }

    a.google-login-button {
        text-decoration: none !important;
    }
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
    try:
        client = OAuth2Session(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, redirect_uri=GOOGLE_REDIRECT_URI, state=st.session_state.get('oauth_state'))
        code = get_query_param('code')  # Fix 1 & 2: Use corrected query params
        token = client.fetch_token(GOOGLE_TOKEN_URL, code=code)
        user_info = requests.get(GOOGLE_USERINFO_URL, headers={'Authorization': f"Bearer {token['access_token']}"}).json()
        if 'error' in user_info:
            st.error(f"Google OAuth error: {user_info['error']}")
            return None
        return user_info
    except Exception as e:
        st.error(f"Error during Google OAuth callback: {str(e)}")
        return None

if st.session_state.page == "Login":
    load_css(st.session_state.theme)
    st.markdown(
        f"""
        <div class="login-card" style="background: {'#2A3B47' if st.session_state.theme == 'dark' else '#FFFFFF'}; border-radius: 15px; padding: 30px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); max-width: 400px; margin: 0 auto; margin-top: 100px;">
        <h1 style="text-align: center; margin-bottom: 20px; font-size: 24px; color: {'#1E90FF' if st.session_state.theme == 'dark' else '#0066CC'}; font-family: 'Roboto', sans-serif;">Welcome to Data Toy AI</h1>
        """,
        unsafe_allow_html=True
    )
    
    username = st.text_input("Username", placeholder="Enter your username", key="username_input")
    password = st.text_input("Password", type="password", placeholder="Enter your password", key="password_input")
    
    if st.button("Login", key="login_button"):
        if verify_user(username, password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.page = "Upload"
            st.session_state.user_info = None
            load_session(username)
            save_auth_state()
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

    # Fix 6: Prevent multiple OAuth callback executions
    if 'code' in st.query_params and not st.session_state.oauth_processed:
        user_info = handle_google_callback()
        if user_info is None:
            st.error("Failed to authenticate with Google. Please try again.")
        else:
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
            st.session_state.oauth_processed = True  # Mark as processed
            load_session(username)
            save_auth_state()
            session_token = st.session_state.session_token
            st.query_params.clear()
            if session_token:
                st.query_params['session_token'] = session_token
            st.rerun()

    if st.button("Sign Up", key="signup_button"):
        st.session_state.page = "Sign Up"
        save_auth_state()
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
    
    new_username = st.text_input("New Username", placeholder="Choose a username", key="new_username_input")
    new_email = st.text_input("Email", placeholder="Enter your email", key="new_email_input")
    new_name = st.text_input("Name", placeholder="Enter your name", key="new_name_input")
    new_password = st.text_input("New Password", type="password", placeholder="Choose a password", key="new_password_input")
    
    if st.button("Register", key="register_button"):
        if add_user(new_username, new_email, new_name, new_password):
            st.success("Registration successful! Please log in.")
            st.session_state.page = "Login"
            save_auth_state()
            st.rerun()
        else:
            st.error("Username already exists. Please choose a different username.")
    
    if st.button("Back to Login", key="back_to_login_button"):
        st.session_state.page = "Login"
        save_auth_state()
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.authenticated:
    load_css(st.session_state.theme)

    def setup_sidebar(logo_path: str = "images/datatoy_logo.png") -> Optional[str]:
        try:
            st.sidebar.image(logo_path, use_column_width=True)
        except FileNotFoundError:
            st.sidebar.markdown("**Data Toy** (Logo not found)", unsafe_allow_html=True)
            st.sidebar.warning(f"Logo file '{logo_path}' not found.")

        if st.session_state.user_info and 'picture' in st.session_state.user_info:
            st.sidebar.image(st.session_state.user_info['picture'], width=100, caption=f"Welcome, {st.session_state.user_info['name']}")
        else:
            st.sidebar.markdown(f"Welcome, {st.session_state.username}")

        st.sidebar.title("Navigation")
        st.sidebar.markdown("<p class='tagline'>Transform your data with AI magic.</p>", unsafe_allow_html=True)

        page = st.sidebar.radio("Go to", ["Upload", "Clean", "Insights", "Visualize", "Predictive", "Share"], key="sidebar_page")
        if page != st.session_state.page:
            st.session_state.page = page
            save_auth_state()
            st.rerun()

        st.sidebar.subheader("Theme")
        theme_choice = st.sidebar.selectbox("Select Theme", ["Dark", "Light"], index=0 if st.session_state.theme == "dark" else 1)
        if theme_choice == "Dark" and st.session_state.theme != "dark":
            st.session_state.theme = "dark"
            save_auth_state()
            st.rerun()
        elif theme_choice == "Light" and st.session_state.theme != "light":
            st.session_state.theme = "light"
            save_auth_state()
            st.rerun()

        st.sidebar.subheader("Your Progress")
        progress_text = ""
        for step, status in st.session_state.progress.items():
            emoji = "✅" if status == "Done" else "🟡" if status == "In Progress" else "⬜"
            progress_text += f"{emoji} {step}: {status}\n"
        st.sidebar.markdown(progress_text)

        if not AI_AVAILABLE:
            st.sidebar.error("⚠️ AI features are disabled. Configure OPENAI_API_KEY.")

        st.sidebar.subheader("AI Data Assistant")
        chat_container = st.sidebar.container()
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(f"**{message['role'].capitalize()}:** {message['content']}")

        chat_input = st.sidebar.chat_input("Ask Data Toy")
        if chat_input:
            df = st.session_state.get('cleaned_df') if st.session_state.get('cleaned_df') is not None else st.session_state.get('df')
            if df is not None:
                st.session_state.chat_history.append({"role": "user", "content": chat_input})
                with st.spinner("Processing your query..."):
                    response = chat_with_gpt(df, chat_input, max_tokens=100)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                save_auth_state()
                st.rerun()
            else:
                st.sidebar.warning("Please upload a dataset first.")

        st.sidebar.markdown("---")
        st.sidebar.markdown("**Feedback**")
        st.sidebar.markdown("[Share your feedback](https://docs.google.com/forms/d/e/1FAIpQLScpUFM0Y5_i5LJDM-HZEZEtOHbLHy4Vp-ek_-819MRZo7Q9rQ/viewform?usp=dialog)")
        st.sidebar.markdown("**Join Our Community**")
        st.sidebar.markdown("[Join our Discord](https://discord.gg/your-invite-link)")
        st.sidebar.markdown("**Upgrade to Premium**")
        st.sidebar.markdown("[Upgrade Now](https://stripe.com/your-checkout-link)")

        is_dev_mode = os.getenv("DEV_MODE") == "true"
        if is_dev_mode:
            st.sidebar.info("Running in DEV_MODE: Unlimited AI suggestions enabled.")

        if st.sidebar.button("Logout"):
            # Fix 5: Fully reset session state with minimal preservation
            preserved = {'username': st.session_state.username}
            st.session_state.clear()
            st.session_state.update(preserved)
            st.session_state.authenticated = False
            st.session_state.page = "Login"
            conn = get_db_connection()
            if conn:
                c = conn.cursor()
                c.execute("DELETE FROM sessions WHERE username = %s", (preserved['username'],))
                conn.commit()
                conn.close()
            st.query_params.clear()
            st.rerun()

        return page

    def main():
        page = setup_sidebar()
        if not page:
            st.error("No page selected.")
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
                    st.warning("Please upload a dataset first.")
                else:
                    render_visualization_page(df)
            elif page == "Predictive":
                df = st.session_state.get('cleaned_df') if st.session_state.get('cleaned_df') is not None else st.session_state.get('df')
                if df is None:
                    st.warning("Please upload a dataset first.")
                else:
                    render_predictive_page(df)
            elif page == "Share":
                st.write("Sharing features coming soon!")
                st.session_state.progress["Share"] = "Done"
        except Exception as e:
            st.error(f"Error rendering {page} page: {str(e)}")
            st.session_state.progress[page] = "Failed"

        save_session(st.session_state.username)

    if __name__ == "__main__":
        main()