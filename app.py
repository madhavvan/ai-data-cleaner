import streamlit as st
# Ensure st.set_page_config is called exactly once at the top
if not hasattr(st, "_is_page_config_set"):
    st.set_page_config(
        page_title="Data ToyAI",
        page_icon="assets/favicon.ico",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st._is_page_config_set = True

import logging
import os
import pickle
import uuid
from logging.handlers import RotatingFileHandler
from typing import Optional

import bcrypt
import psycopg2
import requests

import streamlit.components.v1 as components
from authlib.integrations.requests_client import OAuth2Session
from psycopg2 import sql

try:
    from data_utils import AI_AVAILABLE, chat_with_gpt
    from ui import (render_clean_page, render_insights_page,
                    render_predictive_page, render_upload_page)
    from visualizations import render_visualization_page
except ImportError as e:
    st.error(f"Failed to import required modules (data_utils, ui, visualizations): {e}. Ensure these files are present.")
    st.stop()

try:
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient
    AZURE_KEY_VAULT_AVAILABLE = True
except ImportError:
    AZURE_KEY_VAULT_AVAILABLE = False

# Set up logging with rotation
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    try:
        handler = RotatingFileHandler(
            'app.log',
            maxBytes=5 * 1024 * 1024,
            backupCount=3)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s'))
        logger.addHandler(handler)
        logger.info("Logging initialized.")
    except Exception as e:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')
        logger.error(f"Failed to initialize RotatingFileHandler for app.log: {e}. Using basic logging.")

# Determine the environment (Azure or Streamlit Cloud)
IS_AZURE = os.environ.get("WEBSITE_SITE_NAME") is not None

# Load secrets
GOOGLE_CLIENT_ID = None
GOOGLE_CLIENT_SECRET = None
DB_NAME = None
DB_USER = None
DB_PASSWORD = None
DB_HOST = None
DB_PORT = None
OPENAI_API_KEY = None

if IS_AZURE and AZURE_KEY_VAULT_AVAILABLE:
    logger.info("Running on Azure, attempting to load secrets from Key Vault")
    try:
        key_vault_url = "https://datatoy.vault.azure.net/"
        credential = DefaultAzureCredential()
        secret_client = SecretClient(vault_url=key_vault_url, credential=credential)

        GOOGLE_CLIENT_ID = secret_client.get_secret("GOOGLE-CLIENT-ID").value
        GOOGLE_CLIENT_SECRET = secret_client.get_secret("GOOGLE-CLIENT-SECRET").value
        DB_NAME = secret_client.get_secret("DB-NAME").value
        DB_USER = secret_client.get_secret("DB-USER").value
        DB_PASSWORD = secret_client.get_secret("DB-PASSWORD").value
        DB_HOST = secret_client.get_secret("DB-HOST").value
        DB_PORT = secret_client.get_secret("DB-PORT").value
        OPENAI_API_KEY = secret_client.get_secret("OPENAI-API-KEY").value
        logger.info("Successfully retrieved secrets from Key Vault")
    except Exception as e:
        st.error(f"Failed to retrieve secrets from Key Vault: {str(e)}")
        logger.error(f"Failed to retrieve secrets from Key Vault: {str(e)}")
        st.stop()
else:
    logger.info("Not running on Azure or Key Vault unavailable, falling back to st.secrets or os.environ")
    GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID") or st.secrets.get("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET") or st.secrets.get("GOOGLE_CLIENT_SECRET")
    DB_NAME = os.environ.get("DB_NAME") or st.secrets.get("DB_NAME")
    DB_USER = os.environ.get("DB_USER") or st.secrets.get("DB_USER")
    DB_PASSWORD = os.environ.get("DB_PASSWORD") or st.secrets.get("DB_PASSWORD")
    DB_HOST = os.environ.get("DB_HOST") or st.secrets.get("DB_HOST")
    DB_PORT = os.environ.get("DB_PORT") or st.secrets.get("DB_PORT")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

missing_secrets = [k for k, v in {
    "GOOGLE_CLIENT_ID": GOOGLE_CLIENT_ID,
    "GOOGLE_CLIENT_SECRET": GOOGLE_CLIENT_SECRET,
    "DB_NAME": DB_NAME,
    "DB_USER": DB_USER,
    "DB_PASSWORD": DB_PASSWORD,
    "DB_HOST": DB_HOST,
    "DB_PORT": DB_PORT,
    "OPENAI_API_KEY": OPENAI_API_KEY
}.items() if not v]
if missing_secrets:
    error_msg = f"Missing secrets: {missing_secrets}. Check environment variables, Streamlit secrets, or Key Vault."
    st.error(error_msg)
    logger.error(error_msg)
    st.stop()

# Constants
GOOGLE_REDIRECT_URI = "https://datatoyai.com"
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
SCOPES = ["openid", "email", "profile"]

# Initialize session state
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
if 'oauth_state' not in st.session_state:
    st.session_state.oauth_state = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = []
if 'previous_states' not in st.session_state:
    st.session_state.previous_states = []
if 'redo_states' not in st.session_state:
    st.session_state.redo_states = []
if 'cleaning_history' not in st.session_state:
    st.session_state.cleaning_history = []
if 'cleaning_templates' not in st.session_state:
    st.session_state.cleaning_templates = {}
if 'is_premium' not in st.session_state:
    st.session_state.is_premium = False
if 'ai_suggestions_used' not in st.session_state:
    st.session_state.ai_suggestions_used = 0
if 'dropped_columns' not in st.session_state:
    st.session_state.dropped_columns = []
if 'dashboard_charts' not in st.session_state:
    st.session_state.dashboard_charts = []
if 'dashboard_filters' not in st.session_state:
    st.session_state.dashboard_filters = {}
if 'login_error' not in st.session_state:
    st.session_state.login_error = None
if 'signup_error' not in st.session_state:
    st.session_state.signup_error = None
if 'signup_success' not in st.session_state:
    st.session_state.signup_success = None

# Database Functions
def get_db_connection():
    """Establishes and returns a database connection."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            sslmode="require"
        )
        logger.info("Database connection established.")
        return conn
    except psycopg2.Error as e:
        st.error(f"Failed to connect to database: {e.pgcode} - {e.pgerror}")
        logger.error(f"Database connection failed: {e.pgcode} - {e.pgerror}")
        return None
    except Exception as e:
        st.error(f"Unexpected error connecting to database: {str(e)}")
        logger.error(f"Unexpected error connecting to database: {str(e)}", exc_info=True)
        return None

def init_db():
    """Initializes database tables if they don't exist."""
    logger.info("Initializing database...")
    conn = get_db_connection()
    if conn is None:
        logger.error("Database initialization failed: No connection obtained.")
        st.stop()
        return

    try:
        with conn.cursor() as c:
            logger.debug("Creating 'users' table...")
            c.execute('''CREATE TABLE IF NOT EXISTS users
                         (username TEXT PRIMARY KEY, email TEXT, name TEXT, password BYTEA, google_id TEXT, profile_picture TEXT)''')
            logger.info("Checked/created 'users' table.")

            logger.debug("Checking/creating 'sessions' table...")
            c.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'sessions')")
            table_exists = c.fetchone()[0]
            if table_exists:
                c.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'sessions'")
                columns = [row[0] for row in c.fetchall()]
                if 'username' in columns and 'session_token' in columns and 'session_data' in columns and 'last_accessed' not in columns:
                    c.execute('''CREATE TABLE sessions_new
                                 (session_token TEXT PRIMARY KEY, username TEXT, session_data BYTEA)''')
                    c.execute('''INSERT INTO sessions_new (session_token, username, session_data)
                                 SELECT session_token, username, session_data FROM sessions''')
                    c.execute("DROP TABLE sessions")
                    c.execute("ALTER TABLE sessions_new RENAME TO sessions")
                    logger.info("Migrated 'sessions' table to new schema with session_token as primary key.")
            else:
                c.execute('''CREATE TABLE sessions
                             (session_token TEXT PRIMARY KEY, username TEXT, session_data BYTEA)''')
                logger.info("Created 'sessions' table with session_token as primary key.")

            logger.debug("Creating index 'idx_sessions_username'...")
            c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_username ON sessions (username)")
            logger.info("Checked/created index on 'sessions' (username).")

        conn.commit()
        logger.info("Database initialization complete.")
    except psycopg2.Error as e:
        st.error(f"Database initialization failed: {e.pgcode} - {e.pgerror}")
        logger.error(f"Database initialization failed with psycopg2 error: {e.pgcode} - {e.pgerror}")
        if conn and not conn.closed:
            conn.rollback()
            logger.info("Attempted rollback due to database initialization error.")
        else:
            logger.warning("Rollback skipped: Connection was already closed when exception was caught.")
    except Exception as e:
        st.error(f"Unexpected error during database initialization: {str(e)}")
        logger.error(f"Unexpected error during database initialization: {str(e)}", exc_info=True)
        if conn and not conn.closed:
            conn.rollback()
            logger.info("Attempted rollback due to unexpected error.")
        else:
            logger.warning("Rollback skipped: Connection was already closed when exception was caught.")
    finally:
        if conn and not conn.closed:
            conn.close()
            logger.info("Database connection closed after init.")
        elif conn:
            logger.info("Database connection was already closed before finally block in init_db.")

# Call init_db at the start of the app
init_db()

def add_user(username: str, email: str, name: str, password: str = None,
             google_id: str = None, profile_picture: str = None) -> bool:
    """Add a new user to the database with a hashed password, Google ID, and profile picture."""
    hashed_password = None if password is None else bcrypt.hashpw(
        password.encode('utf-8'), bcrypt.gensalt())
    conn = get_db_connection()
    if conn is None:
        return False

    success = False
    try:
        with conn.cursor() as c:
            c.execute(
                "INSERT INTO users (username, email, name, password, google_id, profile_picture) VALUES (%s, %s, %s, %s, %s, %s)",
                (username, email, name, hashed_password if hashed_password is None else psycopg2.Binary(hashed_password), google_id, profile_picture)
            )
        conn.commit()
        logger.info(f"User added/updated: {username} (Google ID: {google_id})")
        success = True
    except psycopg2.IntegrityError:
        st.session_state['signup_error'] = "Username or email already exists."
        logger.error(f"Integrity error adding/updating user {username}: Username or email already exists")
        if conn and not conn.closed:
            conn.rollback()
    except psycopg2.Error as e:
        st.session_state['signup_error'] = f"Database error: {e.pgcode} - {e.pgerror}"
        logger.error(f"Database error adding/updating user {username}: {str(e)}")
        if conn and not conn.closed:
            conn.rollback()
    except Exception as e:
        st.session_state['signup_error'] = f"An unexpected error occurred during signup."
        logger.error(f"Unexpected error adding/updating user {username}: {str(e)}")
        if conn and not conn.closed:
            conn.rollback()
    finally:
        if conn and not conn.closed:
            conn.close()
    return success

def verify_user(username: str, password: str) -> Optional[dict]:
    """Verifies user credentials and returns user info if valid."""
    conn = get_db_connection()
    if conn is None:
        return None

    user_info = None
    try:
        with conn.cursor() as c:
            c.execute("SELECT username, email, name, password, profile_picture FROM users WHERE username = %s", (username,))
            result = c.fetchone()
        if result:
            db_username, db_email, db_name, stored_password_bytes, db_profile_picture = result
            if stored_password_bytes:
                if isinstance(stored_password_bytes, memoryview):
                    stored_password_bytes = stored_password_bytes.tobytes()
                if isinstance(stored_password_bytes, str):
                    stored_password_bytes = stored_password_bytes.encode('utf-8')
                if bcrypt.checkpw(password.encode('utf-8'), stored_password_bytes):
                    logger.info(f"Password verification successful for user: {username}")
                    user_info = {"username": db_username, "email": db_email, "name": db_name, "picture": db_profile_picture}
                else:
                    logger.warning(f"Password verification failed for user: {username}")
            else:
                logger.warning(f"Login attempt for user {username} failed: No password set (likely Google OAuth user).")
        else:
            logger.warning(f"Login attempt failed: User not found - {username}")
    except Exception as e:
        st.session_state['login_error'] = "An error occurred during login."
        logger.error(f"Error verifying user {username}: {str(e)}")
    finally:
        if conn and not conn.closed:
            conn.close()
    return user_info

def get_user_by_google_id(google_id: str) -> Optional[dict]:
    """Gets user information by Google ID."""
    conn = get_db_connection()
    if conn is None:
        return None

    user_info = None
    try:
        with conn.cursor() as c:
            c.execute("SELECT username, email, name, profile_picture FROM users WHERE google_id = %s", (google_id,))
            result = c.fetchone()
        if result:
            username, email, name, profile_picture = result
            user_info = {"username": username, "email": email, "name": name, "picture": profile_picture, "sub": google_id}
            logger.info(f"Found user by Google ID {google_id}: {username}")
        else:
            logger.info(f"No user found for Google ID: {google_id}")
    except Exception as e:
        logger.error(f"Error fetching user by Google ID {google_id}: {str(e)}")
    finally:
        if conn and not conn.closed:
            conn.close()
    return user_info

# Session Management Functions
def save_auth_state():
    """Saves the current session state to the database."""
    if st.session_state.get('username'):
        logger.debug("Starting save_auth_state")
        if not st.session_state.get('session_token'):
            st.session_state.session_token = str(uuid.uuid4())
            logger.debug(f"Generated new session token in save_auth_state: {st.session_state.session_token}")

        session_data = {
            'authenticated': st.session_state.authenticated,
            'username': st.session_state.username,
            'user_info': st.session_state.user_info,
            'page': st.session_state.page,
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
            'dashboard_filters': st.session_state.get('dashboard_filters')
        }
        session_blob = pickle.dumps(session_data)
        conn = get_db_connection()
        if conn is None:
            logger.debug("Failed to connect to database in save_auth_state")
            return

        try:
            with conn.cursor() as c:
                c.execute(
                    """
                    INSERT INTO sessions (session_token, username, session_data)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (session_token) DO UPDATE SET
                        username = EXCLUDED.username,
                        session_data = EXCLUDED.session_data
                    """,
                    (st.session_state.session_token, st.session_state.username, psycopg2.Binary(session_blob))
                )
            conn.commit()
            logger.info("Session state saved successfully (using session_token as conflict key)")
        except Exception as e:
            logger.error(f"Error in save_auth_state: {str(e)}")
            if conn and not conn.closed:
                conn.rollback()
        finally:
            if conn and not conn.closed:
                conn.close()

def restore_session():
    """Restores the session state from the database using a session token."""
    logger.debug("Starting restore_session")
    session_token = st.session_state.get('session_token') or st.query_params.get('session_token')
    logger.debug(f"Session token from session/query params: {session_token}")
    if session_token:
        conn = get_db_connection()
        if conn is None:
            logger.debug("Failed to connect to database in restore_session")
            return

        try:
            with conn.cursor() as c:
                c.execute(
                    "SELECT username, session_data FROM sessions WHERE session_token = %s",
                    (session_token,)
                )
                result = c.fetchone()
                logger.debug(f"Database query result in restore_session: {'Found' if result else 'Not Found'}")
                if result:
                    username, session_data_blob = result
                    if isinstance(session_data_blob, memoryview):
                        session_data_blob = session_data_blob.tobytes()
                    elif not isinstance(session_data_blob, bytes):
                        raise TypeError(f"Expected bytes/memoryview, got {type(session_data_blob)}")

                    session_data = pickle.loads(session_data_blob)
                    st.session_state.authenticated = session_data.get('authenticated', False)
                    st.session_state.username = username
                    st.session_state.user_info = session_data.get('user_info', None)
                    st.session_state.session_token = session_token
                    st.session_state.page = session_data.get('page', "Upload" if st.session_state.authenticated else "Login")

                    for key, value in session_data.items():
                        if key not in ['authenticated', 'username', 'user_info', 'session_token', 'page']:
                            st.session_state[key] = value
                    logger.info(f"Session restored for user {username}, authenticated: {st.session_state.authenticated}, page: {st.session_state.page}")
                else:
                    logger.debug("No session found for the given session token")
                    st.session_state.authenticated = False
                    st.session_state.session_token = None
                    if 'session_token' in st.query_params:
                        del st.query_params['session_token']
        except Exception as e:
            logger.error(f"Error in restore_session: {str(e)}")
            st.session_state.authenticated = False
            st.session_state.session_token = None
            if 'session_token' in st.query_params:
                del st.query_params['session_token']
        finally:
            if conn and not conn.closed:
                conn.close()
    else:
        logger.debug("No session token found in session state or query parameters")

def save_session(username):
    """Saves the full session state, including authentication."""
    save_auth_state()

def load_session(username):
    """Loads the session state for a user."""
    conn = get_db_connection()
    if conn is None:
        return

    try:
        with conn.cursor() as c:
            c.execute("SELECT session_data FROM sessions WHERE username = %s", (username,))
            result = c.fetchone()

            if result and result[0]:
                session_data_blob = result[0]
                if isinstance(session_data_blob, memoryview):
                    session_data_blob = session_data_blob.tobytes()
                elif not isinstance(session_data_blob, bytes):
                    raise TypeError(f"Expected bytes/memoryview, got {type(session_data_blob)}")

                session_data = pickle.loads(session_data_blob)
                for key, value in session_data.items():
                    if key not in ['authenticated', 'username', 'user_info', 'session_token', 'page']:
                        st.session_state[key] = value
                logger.info(f"Loaded non-auth session data for user {username}")
            else:
                logger.debug(f"No session data found for user {username} in load_session")
    except Exception as e:
        logger.error(f"Error in load_session for user {username}: {str(e)}")
    finally:
        if conn and not conn.closed:
            conn.close()

# UI Functions
def load_css(theme: str = "dark") -> None:
    """Loads CSS styles and apply the appropriate theme class."""
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

    body.dark-theme .stButton#start_cleaning_button > button {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    body.dark-theme .stButton#start_cleaning_button > button:hover {
        background-color: #45a049 !important;
    }
    body.dark-theme .stButton#delete_dataset_button > button {
        background-color: #f44336 !important;
        color: white !important;
    }
    body.dark-theme .stButton#delete_dataset_button > button:hover {
        background-color: #da190b !important;
    }
    """
    components.html(
        f"""
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
        <style>{css}</style>
        <script>
            document.body.className = "{theme}-theme";
        </script>
        """,
        height=0
    )

def render_custom_header(page_title: str) -> None:
    """Render a custom header with the page title."""
    header = st.container()
    with header:
        st.markdown(
            f"<h1 style='margin-top: 20px;'>{page_title}</h1>",
            unsafe_allow_html=True)
    st.markdown(
        "<hr style='border: 1px solid #FFD700;'>",
        unsafe_allow_html=True)

# Google OAuth Functions
def get_google_auth_url():
    """Generates the Google OAuth authorization URL and saves state."""
    base_redirect_uri = GOOGLE_REDIRECT_URI.split('?')[0]
    logger.debug(f"Redirect URI in auth request: {base_redirect_uri}")
    client = OAuth2Session(
        GOOGLE_CLIENT_ID,
        GOOGLE_CLIENT_SECRET,
        redirect_uri=base_redirect_uri,
        scope=SCOPES)
    auth_url, state = client.create_authorization_url(GOOGLE_AUTH_URL)
    st.session_state['oauth_state'] = state
    logger.debug(f"OAuth state saved to session: {state}")
    logger.debug(f"Authorization URL generated: {auth_url}")
    return auth_url

def handle_google_callback():
    """Handles the callback from Google, verifies state, exchanges code."""
    logger.info("--- Handling Google Callback ---")
    callback_error = None
    user_info = None
    query_params_dict = {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in st.query_params.items()}

    callback_state = query_params_dict.get('state')
    saved_state = st.session_state.get('oauth_state')
    logger.info(f"Callback state check: URL State='{callback_state}', Session State='{saved_state}'")

    if not saved_state or callback_state != saved_state:
        callback_error = "OAuth state mismatch. Potential CSRF attack or session issue."
        logger.error(callback_error + f" URL State='{callback_state}', Session State='{saved_state}'")
        if 'state' in st.query_params:
            del st.query_params['state']
        if 'code' in st.query_params:
            del st.query_params['code']
        st.session_state['login_error'] = callback_error
        return None

    if 'oauth_state' in st.session_state:
        del st.session_state['oauth_state']
        logger.info("OAuth state cleared from session after successful validation.")

    code = query_params_dict.get('code')
    logger.debug(f"Authorization code received: {'********' if code else 'None'}")
    if not code:
        callback_error = "No authorization code received from Google"
        logger.error(callback_error)
        if 'state' in st.query_params:
            del st.query_params['state']
        st.session_state['login_error'] = callback_error
        return None

    try:
        base_redirect_uri = GOOGLE_REDIRECT_URI.split('?')[0]
        client = OAuth2Session(
            GOOGLE_CLIENT_ID,
            GOOGLE_CLIENT_SECRET,
            redirect_uri=base_redirect_uri,
            state=saved_state)

        token = client.fetch_token(GOOGLE_TOKEN_URL, code=code)
        logger.info("Successfully fetched OAuth token.")

        user_info_response = requests.get(
            GOOGLE_USERINFO_URL,
            headers={'Authorization': f"Bearer {token['access_token']}"}
        )
        user_info_response.raise_for_status()
        user_info = user_info_response.json()
        logger.info(f"Successfully fetched user info for Google ID: {user_info.get('sub')}")

        if 'code' in st.query_params:
            del st.query_params['code']
        if 'state' in st.query_params:
            del st.query_params['state']
        logger.info("Cleared 'code' and 'state' from query parameters.")

        return user_info

    except Exception as e:
        callback_error = "Error during Google OAuth token exchange/user info fetch."
        if 'invalid_grant' in str(e).lower():
            if 'malformed auth code' in str(e).lower():
                callback_error = "Authentication failed: The authorization code is malformed. Please try signing in again."
            else:
                callback_error = "Authentication failed: The authorization code may have expired or been used. Please try signing in again."
        logger.error(f"Error during Google OAuth token exchange or user info fetch (Code: {'********' if code else 'None'}, State: {saved_state}): {str(e)}")
        if 'invalid_grant' in str(e).lower() or 'malformed auth code' in str(e).lower():
            logger.warning("Detected invalid_grant error, clearing code/state params.")
            if 'code' in st.query_params:
                del st.query_params['code']
            if 'state' in st.query_params:
                del st.query_params['state']
        st.session_state['login_error'] = callback_error
        return None

# Authentication Logic / Page Routing
if not st.session_state.get('authenticated'):
    restore_session()

    if st.query_params.get('code') and st.query_params.get('state'):
        logger.info("Detected Google callback parameters ('code', 'state') on page load.")
        google_user_info = handle_google_callback()

        if google_user_info:
            google_id = google_user_info['sub']
            user_db_info = get_user_by_google_id(google_id)

            username_to_set = None
            email = None
            name = None
            profile_picture = None

            if user_db_info:
                username_to_set, email, name, profile_picture = user_db_info
                logger.info(f"Existing user {username_to_set} found for Google ID {google_id}.")
            else:
                email = google_user_info['email']
                name = google_user_info.get('name', '')
                profile_picture = google_user_info.get('picture')
                username_to_set = email.split('@')[0]

                if add_user(username_to_set, email, name, google_id=google_id, profile_picture=profile_picture):
                    logger.info(f"Successfully added new user {username_to_set} to DB.")
                else:
                    logger.error(f"Failed to add new user {username_to_set} for Google ID {google_id} to DB (likely username exists).")
                    st.session_state['login_error'] = "Failed to register new user account. The username might already exist."
                    username_to_set = None

            if username_to_set:
                st.session_state.authenticated = True
                st.session_state.username = username_to_set
                st.session_state.user_info = google_user_info
                st.session_state.page = "Upload"
                save_auth_state()
                st.rerun()
            else:
                st.session_state.page = "Login"
                st.rerun()
        else:
            st.session_state.page = "Login"
            st.rerun()

    current_page = st.session_state.get('page', 'Login')
    if current_page not in ["Login", "Sign Up"]:
        current_page = "Login"
        st.session_state.page = "Login"

    if current_page == "Login":
        load_css(st.session_state.theme)
        st.markdown(
            f"""
            <div class="login-card" style="background: {'#2A3B47' if st.session_state.theme == 'dark' else '#FFFFFF'}; border-radius: 15px; padding: 30px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); max-width: 400px; margin: 0 auto; margin-top: 100px;">
            <h1 style="text-align: center; margin-bottom: 20px; font-size: 24px; color: {'#1E90FF' if st.session_state.theme == 'dark' else '#0066CC'}; font-family: 'Roboto', sans-serif;">Welcome to Data Toy AI</h1>
            """,
            unsafe_allow_html=True
        )

        if st.session_state.get('login_error'):
            st.error(st.session_state.login_error)
            st.session_state.login_error = None

        if st.session_state.get('signup_success'):
            st.success(st.session_state.signup_success)
            st.session_state.signup_success = None

        login_form = st.form("login_form")
        with login_form:
            username = st.text_input(
                "Username",
                placeholder="Enter your username",
                key="username_input",
                help="Enter your username to log in."
            )
            st.markdown(
                f"""
                <style>
                    #username_input input {{
                        background-color: {'#3C4F5C' if st.session_state.theme == 'dark' else '#F0F4F8'} !important;
                        color: {'#FFFFFF' if st.session_state.theme == 'dark' else '#000000'} !important;
                        border: 1px solid {'#1E90FF' if st.session_state.theme == 'dark' else '#0066CC'} !important;
                        border-radius: 5px !important;
                        padding: 10px !important;
                        font-size: 16px !important;
                        font-family: 'Roboto', sans-serif !important;
                    }}
                    #username_input input:focus {{
                        border-color: {'#FFD700' if st.session_state.theme == 'dark' else '#CC9900'} !important;
                        outline: none !important;
                        box-shadow: 0 0 5px {'rgba(255, 215, 0, 0.5)' if st.session_state.theme == 'dark' else 'rgba(204, 153, 0, 0.5)'} !important;
                    }}
                </style>
                """,
                unsafe_allow_html=True
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                key="password_input",
                help="Enter your password to log in."
            )
            st.markdown(
                f"""
                <style>
                    #password_input input {{
                        background-color: {'#3C4F5C' if st.session_state.theme == 'dark' else '#F0F4F8'} !important;
                        color: {'#FFFFFF' if st.session_state.theme == 'dark' else '#000000'} !important;
                        border: 1px solid {'#1E90FF' if st.session_state.theme == 'dark' else '#0066CC'} !important;
                        border-radius: 5px !important;
                        padding: 10px !important;
                        font-size: 16px !important;
                        font-family: 'Roboto', sans-serif !important;
                    }}
                    #password_input input:focus {{
                        border-color: {'#FFD700' if st.session_state.theme == 'dark' else '#CC9900'} !important;
                        outline: none !important;
                        box-shadow: 0 0 5px {'rgba(255, 215, 0, 0.5)' if st.session_state.theme == 'dark' else 'rgba(204, 153, 0, 0.5)'} !important;
                    }}
                </style>
                """,
                unsafe_allow_html=True
            )

            submitted = st.form_submit_button("Login")
            if submitted:
                user_data = verify_user(username, password)
                if user_data:
                    st.session_state.authenticated = True
                    st.session_state.username = user_data['username']
                    st.session_state.user_info = user_data
                    st.session_state.page = "Upload"
                    save_auth_state()
                    st.rerun()
                else:
                    st.session_state.login_error = "Incorrect username or password."
                    st.rerun()

        st.markdown(
            f"""
            <style>
                #login_button button {{
                    background-color: {'#1E90FF' if st.session_state.theme == 'dark' else '#0066CC'} !important;
                    color: #FFFFFF !important;
                    border: none !important;
                    border-radius: 5px !important;
                    padding: 10px 20px !important;
                    font-size: 16px !important;
                    cursor: pointer !important;
                    transition: background-color 0.3s !important;
                    display: block !important;
                    margin: 10px auto !important;
                    font-family: 'Roboto', sans-serif !important;
                }}
                #login_button button:hover {{
                    background-color: {'#FFD700' if st.session_state.theme == 'dark' else '#CC9900'} !important;
                    color: {'#1C2526' if st.session_state.theme == 'dark' else '#FFFFFF'} !important;
                }}
            </style>
            """,
            unsafe_allow_html=True
        )

        auth_url = get_google_auth_url()
        if auth_url:
            st.markdown(
                f"""
                <a href="{auth_url}" target="_self" style="text-decoration: none;">
                    <div class="google-login-button" style="display: flex; align-items: center; justify-content: center; background-color: #FFFFFF; color: #757575; border: 1px solid #DADCE0; border-radius: 4px; padding: 10px 20px; font-size: 16px; font-family: 'Roboto', sans-serif; font-weight: 500; cursor: pointer; transition: background-color 0.3s ease, box-shadow 0.3s ease; width: 100%; box-sizing: border-box; margin: 10px auto; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);">
                        <img src="https://developers.google.com/identity/images/g-logo.png" alt="Google Icon" style="width: 20px; height: 20px; margin-right: 10px;"/>
                        <span style="color: #757575; font-family: 'Roboto', sans-serif;">Sign in with Google</span>
                    </div>
                </a>
                """,
                unsafe_allow_html=True
            )
        else:
            st.error("Google Login is currently unavailable.")

        st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)
        if st.button("Don't have an account? Sign Up", key="goto_signup"):
            st.session_state.page = "Sign Up"
            st.rerun()

        st.markdown(
            f"""
            <style>
                #goto_signup button {{
                    background-color: {'#1E90FF' if st.session_state.theme == 'dark' else '#0066CC'} !important;
                    color: #FFFFFF !important;
                    border: none !important;
                    border-radius: 5px !important;
                    padding: 10px 20px !important;
                    font-size: 16px !important;
                    cursor: pointer !important;
                    transition: background-color 0.3s !important;
                    display: block !important;
                    margin: 10px auto !important;
                    font-family: 'Roboto', sans-serif !important;
                }}
                #goto_signup button:hover {{
                    background-color: {'#FFD700' if st.session_state.theme == 'dark' else '#CC9900'} !important;
                    color: {'#1C2526' if st.session_state.theme == 'dark' else '#FFFFFF'} !important;
                }}
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

    elif current_page == "Sign Up":
        load_css(st.session_state.theme)
        st.markdown(
            f"""
            <div class="login-card" style="background: {'#2A3B47' if st.session_state.theme == 'dark' else '#FFFFFF'}; border-radius: 15px; padding: 30px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); max-width: 400px; margin: 0 auto; margin-top: 100px;">
            <h1 style="text-align: center; margin-bottom: 20px; font-size: 24px; color: {'#1E90FF' if st.session_state.theme == 'dark' else '#0066CC'}; font-family: 'Roboto', sans-serif;">Sign Up for Data Toy AI</h1>
            """,
            unsafe_allow_html=True
        )

        if st.session_state.get('signup_error'):
            st.error(st.session_state.signup_error)
            st.session_state.signup_error = None

        signup_form = st.form("signup_form")
        with signup_form:
            new_username = st.text_input(
                "New Username",
                placeholder="Choose a username",
                key="new_username_input",
                help="Choose a unique username for your account."
            )
            st.markdown(
                f"""
                <style>
                    #new_username_input input {{
                        background-color: {'#3C4F5C' if st.session_state.theme == 'dark' else '#F0F4F8'} !important;
                        color: {'#FFFFFF' if st.session_state.theme == 'dark' else '#000000'} !important;
                        border: 1px solid {'#1E90FF' if st.session_state.theme == 'dark' else '#0066CC'} !important;
                        border-radius: 5px !important;
                        padding: 10px !important;
                        font-size: 16px !important;
                        font-family: 'Roboto', sans-serif !important;
                    }}
                    #new_username_input input:focus {{
                        border-color: {'#FFD700' if st.session_state.theme == 'dark' else '#CC9900'} !important;
                        outline: none !important;
                        box-shadow: 0 0 5px {'rgba(255, 215, 0, 0.5)' if st.session_state.theme == 'dark' else 'rgba(204, 153, 0, 0.5)'} !important;
                    }}
                </style>
                """,
                unsafe_allow_html=True
            )

            new_email = st.text_input(
                "Email",
                placeholder="Enter your email",
                key="new_email_input",
                help="Enter your email address."
            )
            st.markdown(
                f"""
                <style>
                    #new_email_input input {{
                        background-color: {'#3C4F5C' if st.session_state.theme == 'dark' else '#F0F4F8'} !important;
                        color: {'#FFFFFF' if st.session_state.theme == 'dark' else '#000000'} !important;
                        border: 1px solid {'#1E90FF' if st.session_state.theme == 'dark' else '#0066CC'} !important;
                        border-radius: 5px !important;
                        padding: 10px !important;
                        font-size: 16px !important;
                        font-family: 'Roboto', sans-serif !important;
                    }}
                    #new_email_input input:focus {{
                        border-color: {'#FFD700' if st.session_state.theme == 'dark' else '#CC9900'} !important;
                        outline: none !important;
                        box-shadow: 0 0 5px {'rgba(255, 215, 0, 0.5)' if st.session_state.theme == 'dark' else 'rgba(204, 153, 0, 0.5)'} !important;
                    }}
                </style>
                """,
                unsafe_allow_html=True
            )

            new_name = st.text_input(
                "Name",
                placeholder="Enter your name",
                key="new_name_input",
                help="Enter your full name."
            )
            st.markdown(
                f"""
                <style>
                    #new_name_input input {{
                        background-color: {'#3C4F5C' if st.session_state.theme == 'dark' else '#F0F4F8'} !important;
                        color: {'#FFFFFF' if st.session_state.theme == 'dark' else '#000000'} !important;
                        border: 1px solid {'#1E90FF' if st.session_state.theme == 'dark' else '#0066CC'} !important;
                        border-radius: 5px !important;
                        padding: 10px !important;
                        font-size: 16px !important;
                        font-family: 'Roboto', sans-serif !important;
                    }}
                    #new_name_input input:focus {{
                        border-color: {'#FFD700' if st.session_state.theme == 'dark' else '#CC9900'} !important;
                        outline: none !important;
                        box-shadow: 0 0 5px {'rgba(255, 215, 0, 0.5)' if st.session_state.theme == 'dark' else 'rgba(204, 153, 0, 0.5)'} !important;
                    }}
                </style>
                """,
                unsafe_allow_html=True
            )

            new_password = st.text_input(
                "New Password",
                type="password",
                placeholder="Choose a password",
                key="new_password_input",
                help="Choose a secure password."
            )
            st.markdown(
                f"""
                <style>
                    #new_password_input input {{
                        background-color: {'#3C4F5C' if st.session_state.theme == 'dark' else '#F0F4F8'} !important;
                        color: {'#FFFFFF' if st.session_state.theme == 'dark' else '#000000'} !important;
                        border: 1px solid {'#1E90FF' if st.session_state.theme == 'dark' else '#0066CC'} !important;
                        border-radius: 5px !important;
                        padding: 10px !important;
                        font-size: 16px !important;
                        font-family: 'Roboto', sans-serif !important;
                    }}
                    #new_password_input input:focus {{
                        border-color: {'#FFD700' if st.session_state.theme == 'dark' else '#CC9900'} !important;
                        outline: none !important;
                        box-shadow: 0 0 5px {'rgba(255, 215, 0, 0.5)' if st.session_state.theme == 'dark' else 'rgba(204, 153, 0, 0.5)'} !important;
                    }}
                </style>
                """,
                unsafe_allow_html=True
            )

            confirm_password = st.text_input(
                "Confirm Password*",
                type="password",
                key="signup_confirm_password"
            )
            st.markdown(
                f"""
                <style>
                    #signup_confirm_password input {{
                        background-color: {'#3C4F5C' if st.session_state.theme == 'dark' else '#F0F4F8'} !important;
                        color: {'#FFFFFF' if st.session_state.theme == 'dark' else '#000000'} !important;
                        border: 1px solid {'#1E90FF' if st.session_state.theme == 'dark' else '#0066CC'} !important;
                        border-radius: 5px !important;
                        padding: 10px !important;
                        font-size: 16px !important;
                        font-family: 'Roboto', sans-serif !important;
                    }}
                    #signup_confirm_password input:focus {{
                        border-color: {'#FFD700' if st.session_state.theme == 'dark' else '#CC9900'} !important;
                        outline: none !important;
                        box-shadow: 0 0 5px {'rgba(255, 215, 0, 0.5)' if st.session_state.theme == 'dark' else 'rgba(204, 153, 0, 0.5)'} !important;
                    }}
                </style>
                """,
                unsafe_allow_html=True
            )

            submitted = st.form_submit_button("Register")
            if submitted:
                if not new_username or not new_email or not new_password or not confirm_password:
                    st.session_state['signup_error'] = "Please fill in all required fields (*)."
                elif new_password != confirm_password:
                    st.session_state['signup_error'] = "Passwords do not match."
                elif '@' not in new_email or '.' not in new_email:
                    st.session_state['signup_error'] = "Please enter a valid email address."
                else:
                    if add_user(new_username, new_email, new_name or '', new_password):
                        st.session_state.page = "Login"
                        st.session_state.signup_success = "Registration successful! Please log in."
                        st.session_state.signup_error = None
                        st.rerun()
                    else:
                        st.session_state.signup_error = st.session_state.get('signup_error', "Registration failed. The username or email might already exist.")
                        st.rerun()

        st.markdown(
            f"""
            <style>
                #register_button button {{
                    background-color: {'#1E90FF' if st.session_state.theme == 'dark' else '#0066CC'} !important;
                    color: #FFFFFF !important;
                    border: none !important;
                    border-radius: 5px !important;
                    padding: 10px 20px !important;
                    font-size: 16px !important;
                    cursor: pointer !important;
                    transition: background-color 0.3s !important;
                    display: block !important;
                    margin: 10px auto !important;
                    font-family: 'Roboto', sans-serif !important;
                }}
                #register_button button:hover {{
                    background-color: {'#FFD700' if st.session_state.theme == 'dark' else '#CC9900'} !important;
                    color: {'#1C2526' if st.session_state.theme == 'dark' else '#FFFFFF'} !important;
                }}
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)
        if st.button("Already have an account? Login", key="goto_login"):
            st.session_state.page = "Login"
            st.session_state.signup_error = None
            st.rerun()

        st.markdown(
            f"""
            <style>
                #goto_login button {{
                    background-color: {'#1E90FF' if st.session_state.theme == 'dark' else '#0066CC'} !important;
                    color: #FFFFFF !important;
                    border: none !important;
                    border-radius: 5px !important;
                    padding: 10px 20px !important;
                    font-size: 16px !important;
                    cursor: pointer !important;
                    transition: background-color 0.3s !important;
                    display: block !important;
                    margin: 10px auto !important;
                    font-family: 'Roboto', sans-serif !important;
                }}
                #goto_login button:hover {{
                    background-color: {'#FFD700' if st.session_state.theme == 'dark' else '#CC9900'} !important;
                    color: {'#1C2526' if st.session_state.theme == 'dark' else '#FFFFFF'} !important;
                }}
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

# Main App Logic
if st.session_state.get('authenticated'):
    load_css(st.session_state.theme)

    def setup_sidebar(logo_path: str = "images/datatoy_logo.png") -> Optional[str]:
        """Sets up the sidebar navigation and elements."""
        try:
            if os.path.exists(logo_path):
                st.sidebar.image(logo_path, use_container_width=True)
            else:
                logger.warning(f"Sidebar logo not found at path: {logo_path}")
                st.sidebar.markdown("**Data Toy AI**", unsafe_allow_html=True)
        except Exception as e:
            logger.error(f"Error loading sidebar logo '{logo_path}': {e}")
            st.sidebar.markdown("**Data Toy AI**", unsafe_allow_html=True)

        user_display_name = st.session_state.username
        profile_pic_url = None
        if st.session_state.user_info:
            user_display_name = st.session_state.user_info.get('name', st.session_state.username)
            profile_pic_url = st.session_state.user_info.get('picture')

        if profile_pic_url:
            st.sidebar.image(profile_pic_url, width=80, caption=f"Welcome, {user_display_name}")
        else:
            st.sidebar.markdown(f"Welcome, {user_display_name}")

        st.sidebar.title("Navigation")
        st.sidebar.markdown("<p class='tagline'>Transform your data with AI magic.</p>", unsafe_allow_html=True)

        pages = ["Upload", "Clean", "Insights", "Visualize", "Predictive", "Share"]
        try:
            current_page_index = pages.index(st.session_state.page)
        except ValueError:
            current_page_index = 0
            st.session_state.page = pages[0]

        selected_page = st.sidebar.radio(
            "Go to", pages, index=current_page_index, key="sidebar_nav"
        )

        if selected_page != st.session_state.page:
            st.session_state.page = selected_page
            save_auth_state()
            st.rerun()

        st.sidebar.subheader("Theme")
        theme_options = ["Dark", "Light"]
        current_theme_index = 0 if st.session_state.theme == "dark" else 1
        theme_choice = st.sidebar.selectbox("Select Theme", theme_options, index=current_theme_index, key="theme_select")
        new_theme = theme_choice.lower()
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            save_auth_state()
            st.rerun()

        st.sidebar.subheader("Your Progress")
        progress_text = ""
        for step, status in st.session_state.progress.items():
            emoji = "✅" if status == "Done" else "🟡" if status == "In Progress" else "⬜"
            progress_text += f"{emoji} {step}: {status}\n"
        st.sidebar.markdown(f"```\n{progress_text}\n```")

        if not AI_AVAILABLE:
            st.sidebar.error("⚠️ AI features disabled (OpenAI key missing/invalid).")

        st.sidebar.subheader("AI Data Assistant")
        with st.sidebar.expander("Chat History", expanded=False):
            if not st.session_state.chat_history:
                st.write("No chat history yet.")
            else:
                for message in st.session_state.chat_history:
                    role = message.get("role", "unknown")
                    content = message.get("content", "")
                    with st.chat_message(role):
                        st.write(content)

        chat_input = st.sidebar.chat_input("Ask Data Toy about your data...")
        if chat_input:
            df_context = st.session_state.get('cleaned_df') if st.session_state.get('cleaned_df') is not None else st.session_state.get('df')
            if df_context is not None and AI_AVAILABLE:
                st.session_state.chat_history.append({"role": "user", "content": chat_input})
                with st.spinner("AI Assistant is thinking..."):
                    try:
                        response = chat_with_gpt(df_context, chat_input, max_tokens=150)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as chat_e:
                        logger.error(f"Error calling chat_with_gpt: {chat_e}")
                        st.session_state.chat_history.append({"role": "assistant", "content": "Sorry, I encountered an error trying to respond."})
                save_auth_state()
                st.rerun()
            elif not AI_AVAILABLE:
                st.sidebar.warning("AI Assistant is disabled. Please configure OpenAI API key.")
            else:
                st.sidebar.warning("Please upload or clean a dataset first to use the AI assistant.")

        st.sidebar.markdown("---")
        st.sidebar.markdown("**Feedback & Community**")
        st.sidebar.markdown("- [Share Feedback](https://docs.google.com/forms/d/e/1FAIpQLScpUFM0Y5_i5LJDM-HZEZEtOHbLHy4Vp-ek_-819MRZo7Q9rQ/viewform?usp=dialog)")
        st.sidebar.markdown("- [Join Discord](https://discord.gg/your-invite-link)")
        st.sidebar.markdown("**Support & Upgrade**")
        st.sidebar.markdown("- [Help Documentation](https://your-docs-link.com)")
        st.sidebar.markdown("- [Upgrade to Premium ($5/mo)](https://stripe.com/your-checkout-link)")

        is_dev_mode = os.getenv("DEV_MODE") == "true"
        if is_dev_mode:
            st.sidebar.info("DEV_MODE Active")

        st.sidebar.markdown("---")
        if st.sidebar.button("Logout", key="logout_button"):
            session_token_to_delete = st.session_state.get('session_token')
            keys_to_keep = ['page_config_set']
            current_keys = list(st.session_state.keys())
            for key in current_keys:
                if key not in keys_to_keep:
                    del st.session_state[key]

            st.session_state.authenticated = False
            st.session_state.page = "Login"
            st.session_state.username = None
            st.session_state.user_info = None
            st.session_state.session_token = None

            if session_token_to_delete:
                conn = get_db_connection()
                if conn:
                    try:
                        c = conn.cursor()
                        c.execute("DELETE FROM sessions WHERE session_token = %s", (session_token_to_delete,))
                        conn.commit()
                        logger.info(f"Deleted session for token {session_token_to_delete}")
                    except Exception as del_err:
                        logger.error(f"Error deleting session for token {session_token_to_delete}: {del_err}")
                        if conn and not conn.closed:
                            conn.rollback()
                    finally:
                        if conn and not conn.closed:
                            conn.close()
            st.query_params.clear()
            logger.info(f"User logged out. Session token {session_token_to_delete} cleared.")
            st.rerun()

        return selected_page

    def main() -> None:
        """Main function to render the Data Toy application."""
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
