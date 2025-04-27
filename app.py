import streamlit as st
# Ensure st.set_page_config is called exactly once at the top
if not hasattr(st, "_is_page_config_set"):
    st.set_page_config(
        page_title="Data ToyAI",
        page_icon="assets/favicon.ico", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st._is_page_config_set = True # Use the flag to prevent multiple calls

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

from data_utils import AI_AVAILABLE, chat_with_gpt
from ui import (render_clean_page, render_insights_page,
                render_predictive_page, render_upload_page)
from visualizations import render_visualization_page

# Import Azure Key Vault dependencies only if needed 
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
    # Using basic file handler setup 
    try:
        handler = RotatingFileHandler(
            'app.log',
            maxBytes=5 * 1024 * 1024, # 5MB
            backupCount=3)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')) 
        logger.addHandler(handler)
        # Removed initial logger.info message for brevity if desired, can be added back
    except Exception as log_e:
         logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
         logger.error(f"Failed to initialize RotatingFileHandler for app.log: {log_e}. Using basic logging.")


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
OPENAI_API_KEY = None # Keep this, needed by data_utils import

if IS_AZURE and AZURE_KEY_VAULT_AVAILABLE:
    logger.info("Running on Azure, attempting to load secrets from Key Vault")
    try:
        key_vault_url = "https://datatoy.vault.azure.net/" # Ensure this matches your Key Vault
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

# Validate secrets 
missing_secrets = [k for k, v in {
    "GOOGLE_CLIENT_ID": GOOGLE_CLIENT_ID,
    "GOOGLE_CLIENT_SECRET": GOOGLE_CLIENT_SECRET,
    "DB_NAME": DB_NAME,
    "DB_USER": DB_USER,
    "DB_PASSWORD": DB_PASSWORD,
    "DB_HOST": DB_HOST,
    "DB_PORT": DB_PORT,
    "OPENAI_API_KEY": OPENAI_API_KEY # Keep OpenAI key check if data_utils needs it
}.items() if not v]
if missing_secrets:
    # Modify error message slightly for clarity if needed, but keep structure
    error_msg = f"Missing secrets: {missing_secrets}. Check environment variables, Streamlit secrets, or Key Vault."
    st.error(error_msg)
    logger.error(error_msg)
    st.stop()



# --- Constants 
GOOGLE_REDIRECT_URI = "https://datatoyai.com"
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
SCOPES = ["openid", "email", "profile"]

# --- Session State Initialization 
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
# Add initialization for oauth_state if it wasn't there
if 'oauth_state' not in st.session_state:
    st.session_state.oauth_state = None # Store OAuth state parameter

# --- Database Functions 
def get_db_connection():
    try:
        return psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            sslmode="require" 
        )
    except Exception as e:
        st.error(f"Failed to connect to database: {str(e)}")
        logger.error(f"Failed to connect to database: {str(e)}")
        return None

def init_db():
    # WARNING: This logic for creating/altering the 'sessions' table might be
    #          unreliable or conflict with session token usage.
    conn = get_db_connection()
    if conn is None:
        st.error(
            "Failed to initialize database. Please check your database connection settings.")
        logger.error("Failed to initialize database due to connection failure")
        return 
    # Add try/finally for connection closing (minor robustness improvement)
    try:
        c = conn.cursor()
        # Create users table
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (username TEXT PRIMARY KEY, email TEXT, name TEXT, password BYTEA, google_id TEXT, profile_picture TEXT)''')
        # Check if sessions table exists and has the correct schema 
        c.execute(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'sessions')")
        table_exists = c.fetchone()[0]
        if table_exists:
            # Check if session_token column exists
            c.execute("SELECT EXISTS (SELECT FROM information_schema.columns WHERE table_name = 'sessions' AND column_name = 'session_token')")
            session_token_exists = c.fetchone()[0]
            if not session_token_exists:
                # Add session_token column to existing sessions table
                c.execute("ALTER TABLE sessions ADD COLUMN session_token TEXT")
                logger.debug("Added session_token column to sessions table") 
        else:
            # Create sessions table 
            c.execute('''CREATE TABLE sessions
                         (username TEXT PRIMARY KEY, session_token TEXT, session_data BYTEA)''')
            logger.debug("Created sessions table with session_token column") 
        conn.commit()
        logger.info("Database initialization check complete.") # Added info level for completion
    except Exception as db_err:
        logger.error(f"Error during DB init: {db_err}", exc_info=True) # Log the error
        if conn and not conn.closed: conn.rollback() # Attempt rollback
    finally:
        if conn and not conn.closed: # Ensure close happens
             conn.close()


# Call init_db at the start of the app to ensure the database is initialized
init_db() 


# --- Session Management Functions 
def restore_session():
    logger.debug("Starting restore_session")
    # Check for session token in query parameters
    # Using .get method on query_params directly as it behaves like a dict
    session_token = st.query_params.get('session_token')
    logger.debug(f"Session token from query params: {session_token}")
    if session_token:
        conn = get_db_connection()
        if conn is None:
            logger.debug("Failed to connect to database in restore_session")
            return
        # Add try/finally for connection closing
        try:
            c = conn.cursor()
        
            c.execute(
                "SELECT username, session_data FROM sessions WHERE session_token = %s",
                (session_token,)
            )
            result = c.fetchone()
            logger.debug(f"Database query result in restore_session: {'Found' if result else 'Not Found'}")
            if result:
                username, session_data_blob = result # Renamed variable
                # Ensure blob is bytes
                if isinstance(session_data_blob, memoryview):
                    session_data_blob = session_data_blob.tobytes()
                elif not isinstance(session_data_blob, bytes):
                     raise TypeError(f"Expected bytes/memoryview, got {type(session_data_blob)}")

                session_data = pickle.loads(session_data_blob)
                # Restore authentication state 
                st.session_state.authenticated = session_data.get('authenticated', False)
                st.session_state.username = username
                st.session_state.user_info = session_data.get('user_info', None)
                st.session_state.session_token = session_token
                st.session_state.page = session_data.get('page', "Upload" if st.session_state.authenticated else "Login") # Default depends on auth status

                # Restore other session state variables 
                for key, value in session_data.items():
                    if key not in ['authenticated', 'username',
                                   'user_info', 'session_token', 'page']:
                        st.session_state[key] = value
                logger.info(
                    f"Session restored for user {username}, authenticated: {st.session_state.authenticated}, page: {st.session_state.page}")
            else:
                logger.debug("No session found for the given session token")
                st.session_state.authenticated = False # Log out if token invalid
                st.session_state.session_token = None
                if 'session_token' in st.query_params: del st.query_params['session_token'] # Clean invalid token from URL

        except Exception as e:
            logger.error(f"Error in restore_session: {str(e)}", exc_info=True)
            st.session_state.authenticated = False # Log out on error
            st.session_state.session_token = None
            if 'session_token' in st.query_params: del st.query_params['session_token'] # Clean invalid token from URL
        finally:
            if conn and not conn.closed:
                conn.close()
    else:
        logger.debug("No session token found in query parameters")

def save_auth_state():

    # WARNING: Uses username as conflict key, might be unreliable.
    if st.session_state.get('username'): # Check using .get()
        logger.debug("Starting save_auth_state")
        # Generate a session token if it doesn't exist 
        if not st.session_state.get('session_token'):
            st.session_state.session_token = str(uuid.uuid4())
            logger.debug(f"Generated new session token in save_auth_state: {st.session_state.session_token}")



        # Create session data dict 
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
        # Add try/finally for connection closing
        try:
            c = conn.cursor()
            # Using INSERT ON CONFLICT (username) 
            c.execute("""
                INSERT INTO sessions (username, session_token, session_data)
                VALUES (%s, %s, %s)
                ON CONFLICT (username) DO UPDATE SET
                    session_token = EXCLUDED.session_token,
                    session_data = EXCLUDED.session_data
                """,
                (st.session_state.username, st.session_state.session_token, psycopg2.Binary(session_blob))
             )
            conn.commit()
            logger.info("Session state saved successfully (using username as conflict key)")
        except Exception as e:
            logger.error(f"Error in save_auth_state: {str(e)}", exc_info=True)
            if conn and not conn.closed: conn.rollback() # Attempt rollback
        finally:
            if conn and not conn.closed:
                conn.close()

# Restore session on app startup 
restore_session()

# --- User Management Functions 
def add_user(username: str, email: str, name: str, password: str = None,
             google_id: str = None, profile_picture: str = None):
    """Add a new user to the database with a hashed password, Google ID, and profile picture."""
   
    hashed_password = None if password is None else bcrypt.hashpw(
        password.encode('utf-8'), bcrypt.gensalt())
    conn = get_db_connection()
    if conn is None:
        return False
    # Add try/finally for connection closing
    try:
        c = conn.cursor()
        try: # Inner try for specific integrity error
            c.execute(
                "INSERT INTO users (username, email, name, password, google_id, profile_picture) VALUES (%s, %s, %s, %s, %s, %s)",
                (username, email, name, hashed_password if hashed_password is None else psycopg2.Binary(hashed_password), google_id, profile_picture)
            )
            conn.commit()
            return True # Return True on success
        except psycopg2.IntegrityError:
            # logger.warning(f"Integrity error adding user {username}") # Optional log
            conn.rollback() # Rollback before closing
            return False  # Username or other unique field already exists
        except Exception as e_inner:
            logger.error(f"DB error during user insert for {username}: {e_inner}", exc_info=True)
            conn.rollback()
            return False
    finally:
        if conn and not conn.closed:
            conn.close()


def verify_user(username: str, password: str) -> bool:
    """Verify user credentials."""
    
    conn = get_db_connection()
    if conn is None:
        return False
    # Add try/finally for connection closing
    try:
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username = %s", (username,))
        result = c.fetchone()

        if result and result[0] is not None: # Check if stored_password exists and is not NULL
            stored_password = result[0]
            if isinstance(stored_password, memoryview):
                stored_password = stored_password.tobytes()
          
            if isinstance(stored_password, str):
                stored_password = stored_password.encode('utf-8')

            # Ensure it's bytes before checking
            if isinstance(stored_password, bytes):
                 try:
                    return bcrypt.checkpw(password.encode('utf-8'), stored_password)
                 except Exception as bcrypt_e: # Catch potential bcrypt errors
                     logger.error(f"Error during bcrypt check for {username}: {bcrypt_e}")
                     return False
            else:
                 logger.warning(f"Stored password for {username} is not bytes after conversion.")
                 return False
        else:
            # User not found or password is NULL
            return False
    except Exception as e:
        logger.error(f"Error during user verification for {username}: {e}", exc_info=True)
        return False
    finally:
        if conn and not conn.closed:
            conn.close()


def get_user_by_google_id(google_id: str):
    """Get user by Google ID."""
    
    conn = get_db_connection()
    if conn is None:
        return None
    # Add try/finally for connection closing
    try:
        c = conn.cursor()
        c.execute(
            "SELECT username, email, name, profile_picture FROM users WHERE google_id = %s",
            (google_id,)
        )
        result = c.fetchone()
        return result # Return tuple or None
    except Exception as e:
        logger.error(f"Error fetching user by google_id {google_id}: {e}", exc_info=True)
        return None
    finally:
        if conn and not conn.closed:
            conn.close()


# These might be redundant given restore_session and save_auth_state but keeping for fidelity
def save_session(username):
    # Save the full session state, including authentication
    save_auth_state()

def load_session(username):
    conn = get_db_connection()
    if conn is None:
        return
    # Add try/finally for connection closing
    try:
        c = conn.cursor()
        c.execute("SELECT session_data FROM sessions WHERE username = %s", (username,))
        result = c.fetchone()

        if result and result[0]: # Check result is not None and blob is not None
            session_data_blob = result[0]
            # Ensure blob is bytes
            if isinstance(session_data_blob, memoryview):
                session_data_blob = session_data_blob.tobytes()
            elif not isinstance(session_data_blob, bytes):
                raise TypeError(f"Expected bytes/memoryview, got {type(session_data_blob)}")

            session_data = pickle.loads(session_data_blob)
            for key, value in session_data.items():
                if key not in ['authenticated', 'username', 'user_info',
                               'session_token', 'page']:
                    st.session_state[key] = value
            logger.info(f"Loaded non-auth session data for user {username}")
        else:
            logger.debug(f"No session data found for user {username} in load_session")
    except Exception as e:
        logger.error(f"Error in load_session for user {username}: {e}", exc_info=True)
    finally:
         if conn and not conn.closed:
            conn.close()

# --- UI Functions 
def load_css(theme: str = "dark") -> None:
    """Loads CSS styles and apply the appropriate theme class."""
    
    # Ensure ALL rules are included
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

    body.dark-theme .css-1d391kg { /* Sidebar selector might change with Streamlit versions */
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

    body.light-theme .css-1d391kg { /* Sidebar selector might change */
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

    /* Specific button styles from app (3).py */
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
    # Use components.html to inject the CSS and set the body class immediately
    components.html(
        f"""
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
        <style>
            {css}
        </style>
        <script>
            document.body.className = "{theme}-theme";
            // console.log("Applied body class:", document.body.className); // From app (3).py
            // Debugging line from app (3).py
            // document.body.style.backgroundColor = "{'#1C2526' if theme == 'dark' else '#F0F4F8'}"; // Commented out if not needed
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

# --- Google OAuth Functions ---

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
    st.session_state['oauth_state'] = state # Save state to session
    logger.debug(f"OAuth state saved to session: {state}")
    logger.debug(f"Authorization URL generated: {auth_url}")
    return auth_url

# handle_google_callback MODIFIED to add state check
def handle_google_callback():
    """Handles the callback from Google, verifies state, exchanges code."""
    # *** THIS FUNCTION IS MODIFIED TO ADD STATE CHECK ***
    logger.info("--- Handling Google Callback ---")
    callback_error = None
    user_info = None
    # Use .get() on query_params which is now dictionary-like
    # Convert list values to single values if applicable
    query_params_dict = {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in st.query_params.items()}


    # 1. State Check (This is the crucial added security step)
    callback_state = query_params_dict.get('state')
    saved_state = st.session_state.get('oauth_state')
    logger.info(f"Callback state check: URL State='{callback_state}', Session State='{saved_state}'")

    # --- STATE CHECK ---
    if not saved_state or callback_state != saved_state:
        callback_error = "OAuth state mismatch. Potential CSRF attack or session issue."
        logger.error(callback_error + f" URL State='{callback_state}', Session State='{saved_state}'")
        # IMPORTANT: If this error occurs frequently, it indicates the session state (oauth_state)
        #            is being lost during the redirect. The session logic in restore_session/save_auth_state
        #            might need to be replaced with a more robust database-token approach.
        # Clean up URL parameters to prevent loops
        if 'state' in st.query_params: del st.query_params['state']
        if 'code' in st.query_params: del st.query_params['code']
        st.session_state['login_error'] = callback_error # Store error for login page
        return None # Indicate failure

    # If state check passed, clear the state from session as it's used
    if 'oauth_state' in st.session_state:
        del st.session_state['oauth_state']
        logger.info("OAuth state cleared from session after successful validation.")

    # 2. Code Exchange 
    code = query_params_dict.get('code')
    # Mask code in log for security
    logger.debug(f"Authorization code received: {'********' if code else 'None'}")
    if not code:
        callback_error = "No authorization code received from Google"
        logger.error(callback_error)
        # State might still be in URL if code is missing, clear it
        if 'state' in st.query_params: del st.query_params['state']
        st.session_state['login_error'] = callback_error
        return None # Indicate failure

    try:
        base_redirect_uri = GOOGLE_REDIRECT_URI.split('?')[0]
        # We use the validated saved_state here for the OAuth2Session
        client = OAuth2Session(
            GOOGLE_CLIENT_ID,
            GOOGLE_CLIENT_SECRET,
            redirect_uri=base_redirect_uri,
            state=saved_state) # Use the state we confirmed matches

        # REMINDER: Check Google Cloud Console Redirect URIs and Client ID/Secret
        # if you get 'invalid_grant' errors here.
        token = client.fetch_token(GOOGLE_TOKEN_URL, code=code)
        logger.info("Successfully fetched OAuth token.")

        user_info_response = requests.get(
            GOOGLE_USERINFO_URL, headers={'Authorization': f"Bearer {token['access_token']}"}
        )
        user_info_response.raise_for_status() # Check for HTTP errors (4xx, 5xx)
        user_info = user_info_response.json()
        logger.info(f"Successfully fetched user info for Google ID: {user_info.get('sub')}")

        # Clear URL parameters on success BEFORE returning
        logger.info("Clearing code and state from query parameters after successful token exchange.")
        if 'code' in st.query_params: del st.query_params['code']
        if 'state' in st.query_params: del st.query_params['state']

        return user_info # Success

    except Exception as e:
        callback_error = f"Error during Google OAuth token exchange/user info fetch: {str(e)}"
        logger.error(callback_error, exc_info=True)
        st.session_state['login_error'] = callback_error # Store error for login page

        # If invalid_grant, clear the likely bad code/state as a precaution
        if 'invalid_grant' in str(e).lower() or 'malformed auth code' in str(e).lower():
             logger.warning("Detected invalid_grant error, clearing code/state params.")
             if 'code' in st.query_params: del st.query_params['code']
             if 'state' in st.query_params: del st.query_params['state']
        return None # Indicate failure


# --- Authentication Logic / Page Routing 
# Check authentication status (might be True if restore_session worked)
if not st.session_state.get('authenticated'):

    # --- Google Callback Handling ---
    # Check for callback parameters right away
    # Using .get for query_params directly
    if st.query_params.get('code') and st.query_params.get('state'):
        logger.info("Detected Google callback parameters ('code', 'state') on page load.")
        google_user_info = handle_google_callback() # Call the modified handler

        if google_user_info:
            # If callback successful, process user info
            logger.info("Google callback handled successfully, processing user info.")
            google_id = google_user_info['sub']
            user_db_info = get_user_by_google_id(google_id) # Function handles DB connection

            username_to_set = None
            email = None
            name = None
            profile_picture = None

            if user_db_info:
                # Existing user found in DB (unpack tuple)
                username_to_set, email, name, profile_picture = user_db_info
                logger.info(f"Existing user {username_to_set} found for Google ID {google_id}.")
            else:
                # New user - Create based on Google info
                logger.info(f"New user registration via Google for ID {google_id}.")
                email = google_user_info['email']
                name = google_user_info.get('name', '')
                profile_picture = google_user_info.get('picture')
                # Create a simple username 
                username_to_set = email.split('@')[0] # Using original logic

                # Add user to DB
                if add_user(username_to_set, email, name, google_id=google_id, profile_picture=profile_picture):
                    logger.info(f"Successfully added new user {username_to_set} to DB.")
                else:
                    logger.error(f"Failed to add new user {username_to_set} for Google ID {google_id} to DB (likely username exists).")
                    st.error("Failed to register new user account. The username might already exist.")
                    username_to_set = None # Prevent login

            # If we have a username (either existing or newly created/added)
            if username_to_set:
                st.session_state.authenticated = True
                st.session_state.username = username_to_set
                st.session_state.user_info = google_user_info # Store Google info
                st.session_state.page = "Upload"
                # load_session(username_to_set) # Call original load_session from app (3).py
                save_auth_state() # Save the new auth state (will also generate/save token)

                # Update URL with the session token generated by save_auth_state
                session_token = st.session_state.get('session_token')
                # Params should have been cleared by handle_google_callback on success
                if session_token:
                    st.query_params['session_token'] = session_token
                    logger.info("Added session_token to query params.")
                st.rerun() # Rerun to show the main app page
            else:
                # Failed login due to DB error or other issue during user processing
                st.session_state.page = "Login"
                # Error message might have been set by add_user
                st.rerun() # Rerun to show login page again
        else:
            # handle_google_callback failed and set an error message in session_state
            # Flow continues to render Login page below, which should display the error.
            st.session_state.page = "Login"
            logger.warning("Google callback detected but handle_google_callback failed.")

    # --- Render Login or Sign Up page if not authenticated ---
   
    current_page = st.session_state.get('page', 'Login')
    if current_page not in ["Login", "Sign Up"]:
        current_page = "Login"
        st.session_state.page = "Login"

    if current_page == "Login":
        load_css(st.session_state.theme)
        st.markdown(
            f"""<div class="login-card" style="...">...</div>""", unsafe_allow_html=True
        )
        # Display login errors (e.g., from failed Google callback or bad password)
        if st.session_state.get('login_error'):
            st.error(st.session_state.login_error)
            st.session_state.login_error = None # Clear after displaying

        username = st.text_input("Username", placeholder="Enter your username", key="username_input", help="...")
        st.markdown(f"""<style>#username_input input {{...}}</style>""", unsafe_allow_html=True) # Keep full style
        password = st.text_input("Password", type="password", placeholder="Enter your password", key="password_input", help="...")
        st.markdown(f"""<style>#password_input input {{...}}</style>""", unsafe_allow_html=True) # Keep full style

        if st.button("Login", key="login_button", help="..."):
            if verify_user(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.page = "Upload"
                st.session_state.user_info = None # Clear Google info
                # load_session(username) 
                save_auth_state() # Save authentication state (generates/updates token)
                # Update URL params
                session_token = st.session_state.get('session_token')
                st.query_params.clear()
                if session_token:
                    st.query_params['session_token'] = session_token
                st.rerun()
            else:
                st.error("Incorrect username or password")
        st.markdown(f"""<style>#login_button button {{...}}</style>""", unsafe_allow_html=True) # Keep full style

        # Google Login Button 
        auth_url = get_google_auth_url() # Calls the original function which now saves state
        st.markdown(f"""<a href="{auth_url}" target="_self" style="...">...</a>""", unsafe_allow_html=True) # Keep full markdown/style

    

        # Sign Up button 
        if st.button("Sign Up", key="signup_button", help="..."):
            st.session_state.page = "Sign Up"
            # save_auth_state() # Removed save here to match previous version behaviour
            st.rerun()
        st.markdown(f"""<style>#signup_button button {{...}}</style>""", unsafe_allow_html=True) # Keep full style

        st.markdown('</div>', unsafe_allow_html=True) # Close login-card

    elif current_page == "Sign Up":

        load_css(st.session_state.theme)
        st.markdown(f"""<div class="login-card" style="...">...</div>""", unsafe_allow_html=True)
        new_username = st.text_input("New Username", placeholder="Choose a username", key="new_username_input", help="...")
        st.markdown(f"""<style>#new_username_input input {{...}}</style>""", unsafe_allow_html=True) # Keep full style
        new_email = st.text_input("Email", placeholder="Enter your email", key="new_email_input", help="...")
        st.markdown(f"""<style>#new_email_input input {{...}}</style>""", unsafe_allow_html=True) # Keep full style
        new_name = st.text_input("Name", placeholder="Enter your name", key="new_name_input", help="...")
        st.markdown(f"""<style>#new_name_input input {{...}}</style>""", unsafe_allow_html=True) # Keep full style
        new_password = st.text_input("New Password", type="password", placeholder="Choose a password", key="new_password_input", help="...")
        st.markdown(f"""<style>#new_password_input input {{...}}</style>""", unsafe_allow_html=True) # Keep full style

        if st.button("Register", key="register_button", help="..."):
            if add_user(new_username, new_email, new_name, new_password):
                st.success("Registration successful! Please log in.")
                st.session_state.page = "Login"
                # save_auth_state() # Removed save here
                st.rerun()
            else:
                # Error message improved slightly for clarity
                st.error("Username already exists or another registration error occurred.")
        st.markdown(f"""<style>#register_button button {{...}}</style>""", unsafe_allow_html=True) # Keep full style

        if st.button("Back to Login", key="back_to_login_button", help="..."):
            st.session_state.page = "Login"
            # save_auth_state() # Removed save here
            st.rerun()
        st.markdown(f"""<style>#back_to_login_button button {{...}}</style>""", unsafe_allow_html=True) # Keep full style

        st.markdown('</div>', unsafe_allow_html=True) # Close login-card



if st.session_state.get('authenticated'): # Use .get() for safety
    load_css(st.session_state.theme)


    def setup_sidebar(logo_path: str = "images/datatoy_logo.png") -> Optional[str]:
        # Ensure logo path and error handling are identical
        try:
            st.sidebar.image(logo_path, use_container_width=True)
        except FileNotFoundError: # Keep exact exception type if specified
             st.sidebar.markdown("**Data Toy** (Logo not found)", unsafe_allow_html=True)
             st.sidebar.warning(f"Logo file '{logo_path}' not found. Please add it to the project directory.")
        # ... rest of the user info display logic ...
        # ... rest of the Navigation title/tagline ...
        page = st.sidebar.radio("Go to", ["Upload", "Clean", "Insights", "Visualize", "Predictive", "Share"], key="sidebar_page", index=["Upload", "Clean", "Insights", "Visualize", "Predictive", "Share"].index(st.session_state.page))
        if page != st.session_state.page:
            st.session_state.page = page
            save_auth_state()
            st.rerun()
        # ... rest of Theme toggle logic ...
        # ... rest of Progress tracker logic ...
        # ... rest of AI Assistant logic ...
        # ... rest of Feedback/Community/Upgrade links ...
        # ... rest of Dev Mode indicator ...
        # ... rest of Logout button logic (ensure it deletes session by username as in app (3).py) ...
        if st.sidebar.button("Logout"):
            username_to_delete = st.session_state.get('username') # Get username before clearing state
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.user_info = None
            st.session_state.session_token = None
            st.session_state.page = "Login"
            # Clear session data from the database using username (as per app (3).py)
            if username_to_delete: # Check if username exists before trying delete
                conn = get_db_connection()
                if conn:
                    # Add try/finally
                    try:
                        c = conn.cursor()
                        c.execute("DELETE FROM sessions WHERE username = %s", (username_to_delete,))
                        conn.commit()
                        logger.info(f"Deleted session for user {username_to_delete}")
                    except Exception as del_err:
                         logger.error(f"Error deleting session for {username_to_delete}: {del_err}", exc_info=True)
                         if conn and not conn.closed: conn.rollback()
                    finally:
                         if conn and not conn.closed: conn.close()
            st.query_params.clear()
            st.rerun()

        return page # Return page from setup_sidebar


    def main() -> None:
        """Main function to render the Data Toy application."""
        page = setup_sidebar()

        if not page:

            st.error("No page selected. Please select a page from the sidebar.")
            return

        page_titles = {
            "Upload": "Upload Your Dataset", "Clean": "Clean Your Dataset",
            "Insights": "Insights Dashboard", "Visualize": "Visualize Your Dataset",
            "Predictive": "Predictive Analytics", "Share": "Share Your Work"
        }
        render_custom_header(page_titles.get(page, "Data Toy")) 

        try: 
            if page == "Upload":
                render_upload_page()
            elif page == "Clean":
                render_clean_page() # Assume this handles missing df internally if needed
            elif page == "Insights":
                render_insights_page() # Assume this handles missing df internally
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

        # Save session on every interaction
        save_session(st.session_state.username)

    if __name__ == "__main__":
        main()
