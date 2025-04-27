import streamlit as st
# Ensure st.set_page_config is called exactly once at the top
# Use a check to prevent calling it multiple times during reruns
if 'page_config_set' not in st.session_state:
    st.set_page_config(
        page_title="Data ToyAI",
        page_icon="assets/favicon.ico", # Make sure this path is correct
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.session_state['page_config_set'] = True

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

# --- Attempt to import local modules ---
# These imports assume data_utils.py, ui.py, and visualizations.py are in the same directory or accessible in the Python path.
try:
    from data_utils import AI_AVAILABLE, chat_with_gpt
    from ui import (render_clean_page, render_insights_page,
                    render_predictive_page, render_upload_page)
    from visualizations import render_visualization_page
except ImportError as e:
    st.error(f"Failed to import required modules (data_utils, ui, visualizations): {e}. Ensure these files are present.")
    st.stop()
# --- End of local module imports ---

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
    log_file = 'app.log'
    try:
        handler = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024, # 5MB
            backupCount=3)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')) # Added funcName
        logger.addHandler(handler)
        logger.info("Logging initialized.")
    except Exception as e:
        # Fallback to basic logging if file rotation fails (e.g., permissions)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')
        logger.error(f"Failed to initialize RotatingFileHandler for {log_file}: {e}. Using basic logging.")


# Determine the environment (Azure or Streamlit Cloud)
IS_AZURE = os.environ.get("WEBSITE_SITE_NAME") is not None  # Azure App Service sets this

# --- Load secrets ---
# Wrapped loading in a function for clarity
def load_secrets():
    if IS_AZURE and AZURE_KEY_VAULT_AVAILABLE:
        logger.info("Running on Azure, attempting to load secrets from Key Vault")
        try:
            key_vault_url = "https://datatoy.vault.azure.net/" # Ensure this is your correct Key Vault URL
            credential = DefaultAzureCredential()
            secret_client = SecretClient(vault_url=key_vault_url, credential=credential)

            secrets = {
                "GOOGLE_CLIENT_ID": secret_client.get_secret("GOOGLE-CLIENT-ID").value,
                "GOOGLE_CLIENT_SECRET": secret_client.get_secret("GOOGLE-CLIENT-SECRET").value,
                "DB_NAME": secret_client.get_secret("DB-NAME").value,
                "DB_USER": secret_client.get_secret("DB-USER").value,
                "DB_PASSWORD": secret_client.get_secret("DB-PASSWORD").value,
                "DB_HOST": secret_client.get_secret("DB-HOST").value,
                "DB_PORT": secret_client.get_secret("DB-PORT").value,
                "OPENAI_API_KEY": secret_client.get_secret("OPENAI-API-KEY").value
            }
            logger.info("Successfully retrieved secrets from Key Vault")
            return secrets
        except Exception as e:
            st.error(f"Failed to retrieve secrets from Key Vault: {str(e)}")
            logger.error(f"Failed to retrieve secrets from Key Vault: {str(e)}")
            st.stop()
            return {} # Should not be reached due to st.stop()
    else:
        logger.info("Not running on Azure or Key Vault unavailable, falling back to st.secrets or os.environ")
        secrets = {
            "GOOGLE_CLIENT_ID": os.environ.get("GOOGLE_CLIENT_ID") or st.secrets.get("GOOGLE_CLIENT_ID"),
            "GOOGLE_CLIENT_SECRET": os.environ.get("GOOGLE_CLIENT_SECRET") or st.secrets.get("GOOGLE_CLIENT_SECRET"),
            "DB_NAME": os.environ.get("DB_NAME") or st.secrets.get("DB_NAME"),
            "DB_USER": os.environ.get("DB_USER") or st.secrets.get("DB_USER"),
            "DB_PASSWORD": os.environ.get("DB_PASSWORD") or st.secrets.get("DB_PASSWORD"),
            "DB_HOST": os.environ.get("DB_HOST") or st.secrets.get("DB_HOST"),
            "DB_PORT": os.environ.get("DB_PORT") or st.secrets.get("DB_PORT"),
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
        }
        return secrets

app_secrets = load_secrets()

# Validate secrets
missing_secrets = [k for k, v in app_secrets.items() if not v]
if missing_secrets:
    error_msg = f"Missing secrets: {missing_secrets}. Check environment variables, Streamlit secrets, or Azure Key Vault."
    st.error(error_msg)
    logger.error(error_msg)
    st.stop()

# Assign secrets to global constants (use .get for safety, though validated above)
GOOGLE_CLIENT_ID = app_secrets.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = app_secrets.get("GOOGLE_CLIENT_SECRET")
DB_NAME = app_secrets.get("DB_NAME")
DB_USER = app_secrets.get("DB_USER")
DB_PASSWORD = app_secrets.get("DB_PASSWORD")
DB_HOST = app_secrets.get("DB_HOST")
DB_PORT = app_secrets.get("DB_PORT")
OPENAI_API_KEY = app_secrets.get("OPENAI_API_KEY")
# --- End of secrets loading ---


# --- OAuth Configuration ---
GOOGLE_REDIRECT_URI = "https://datatoyai.com" # Your registered Redirect URI
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
SCOPES = ["openid", "email", "profile"]
# --- End of OAuth Configuration ---


# --- Initialize session state ---
# Use functions to check and initialize to avoid clutter
def init_session_state_key(key, default_value):
    if key not in st.session_state:
        st.session_state[key] = default_value

init_session_state_key('chat_history', [])
init_session_state_key('theme', "dark")
init_session_state_key('authenticated', False)
init_session_state_key('username', None)
init_session_state_key('page', "Login")
init_session_state_key('progress', {
    "Upload": "Not Started", "Clean": "Not Started", "Insights": "Not Started",
    "Visualize": "Not Started", "Predictive": "Not Started", "Share": "Not Started"
})
init_session_state_key('user_info', None) # Stores Google user info if logged in via Google
init_session_state_key('session_token', None) # For persistent sessions
init_session_state_key('oauth_state', None) # Store OAuth state parameter
init_session_state_key('df', None) # Uploaded DataFrame
init_session_state_key('cleaned_df', None) # Cleaned DataFrame
init_session_state_key('logs', []) # Cleaning logs
init_session_state_key('suggestions', []) # AI Cleaning Suggestions
init_session_state_key('previous_states', []) # For undo/redo
init_session_state_key('redo_states', []) # For undo/redo
init_session_state_key('cleaning_history', [])
init_session_state_key('cleaning_templates', {})
init_session_state_key('is_premium', False) # Example premium status
init_session_state_key('ai_suggestions_used', 0)
init_session_state_key('dropped_columns', [])
init_session_state_key('dashboard_charts', [])
init_session_state_key('dashboard_filters', {})
init_session_state_key('login_error', None) # To display login errors
init_session_state_key('signup_error', None) # To display signup errors
init_session_state_key('signup_success', None) # To display signup success message
# --- End of session state initialization ---


# --- Database Functions ---
@st.cache_resource # Cache the connection pool
def get_db_connection():
    """Establishes and returns a database connection."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            sslmode="require" # Common requirement for cloud databases
        )
        logger.info("Database connection established.")
        return conn
    except Exception as e:
        st.error(f"Failed to connect to database: {str(e)}")
        logger.error(f"Database connection failed: {str(e)}")
        return None

def init_db():
    """Initializes database tables if they don't exist."""
    logger.info("Initializing database...")
    conn = get_db_connection()
    if conn is None:
        st.error("Database initialization failed: No connection.")
        logger.error("Database initialization failed: No connection.")
        st.stop() # Stop execution if DB cannot be initialized
        return

    try:
        with conn.cursor() as c:
            # Create users table
            c.execute('''CREATE TABLE IF NOT EXISTS users (
                         username TEXT PRIMARY KEY,
                         email TEXT UNIQUE,
                         name TEXT,
                         password BYTEA,
                         google_id TEXT UNIQUE,
                         profile_picture TEXT
                       )''')
            logger.info("Checked/created 'users' table.")

            # Create sessions table (Simplified: storing pickled session state)
            # Consider security implications of storing pickled data.
            # A more robust solution might use a dedicated session management library or store specific fields.
            c.execute('''CREATE TABLE IF NOT EXISTS sessions (
                         session_token TEXT PRIMARY KEY,
                         username TEXT REFERENCES users(username) ON DELETE CASCADE,
                         session_data BYTEA,
                         last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                       )''')
            logger.info("Checked/created 'sessions' table.")

            # Add index for faster session lookup by username
            c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_username ON sessions (username)")
            logger.info("Checked/created index on 'sessions'.")

            # Add index for faster session lookup by token
            c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions (session_token)")
            logger.info("Checked/created index on 'sessions'.")

        conn.commit()
        logger.info("Database initialization complete.")
    except Exception as e:
        st.error(f"Database initialization error: {str(e)}")
        logger.error(f"Database initialization error: {str(e)}")
        conn.rollback() # Rollback changes on error
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed after init.")

# Call init_db at the start of the app
init_db()

def add_user(username: str, email: str, name: str, password: str = None,
             google_id: str = None, profile_picture: str = None) -> bool:
    """Adds or updates a user in the database. Hashes password if provided."""
    hashed_password = None
    if password:
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    conn = get_db_connection()
    if conn is None: return False

    success = False
    try:
        with conn.cursor() as c:
            # Use ON CONFLICT to handle potential updates (e.g., linking Google ID to existing user)
            # Assumes username or google_id should be unique identifiers
            query = sql.SQL("""
                INSERT INTO users (username, email, name, password, google_id, profile_picture)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (username) DO UPDATE SET
                    email = EXCLUDED.email,
                    name = EXCLUDED.name,
                    password = COALESCE(EXCLUDED.password, users.password),
                    google_id = COALESCE(EXCLUDED.google_id, users.google_id),
                    profile_picture = COALESCE(EXCLUDED.profile_picture, users.profile_picture)
                ON CONFLICT (google_id) WHERE google_id IS NOT NULL DO UPDATE SET
                    username = EXCLUDED.username,
                    email = EXCLUDED.email,
                    name = EXCLUDED.name,
                    password = COALESCE(EXCLUDED.password, users.password),
                    profile_picture = COALESCE(EXCLUDED.profile_picture, users.profile_picture)
            """)
            c.execute(query, (username, email, name, hashed_password, google_id, profile_picture))
        conn.commit()
        logger.info(f"User added/updated: {username} (Google ID: {google_id})")
        success = True
    except psycopg2.Error as e:
        st.session_state['signup_error'] = f"Database error: {e.pgcode} - {e.pgerror}"
        logger.error(f"Database error adding/updating user {username}: {str(e)}")
        conn.rollback()
    except Exception as e:
        st.session_state['signup_error'] = f"An unexpected error occurred during signup."
        logger.error(f"Unexpected error adding/updating user {username}: {str(e)}")
        conn.rollback()
    finally:
        if conn:
            conn.close()
    return success

def verify_user(username: str, password: str) -> Optional[dict]:
    """Verifies user credentials and returns user info if valid."""
    conn = get_db_connection()
    if conn is None: return None

    user_info = None
    try:
        with conn.cursor() as c:
            c.execute("SELECT username, email, name, password, profile_picture FROM users WHERE username = %s", (username,))
            result = c.fetchone()
        if result:
            db_username, db_email, db_name, stored_password_bytes, db_profile_picture = result
            if stored_password_bytes: # Check if password exists (not a Google-only user)
                 # Convert memoryview to bytes if needed
                if isinstance(stored_password_bytes, memoryview):
                    stored_password_bytes = stored_password_bytes.tobytes()

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
        if conn:
            conn.close()
    return user_info


def get_user_by_google_id(google_id: str) -> Optional[dict]:
    """Gets user information by Google ID."""
    conn = get_db_connection()
    if conn is None: return None

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
        if conn:
            conn.close()
    return user_info
# --- End of Database Functions ---


# --- Session Management Functions ---
def save_session_state_to_db():
    """Saves the current Streamlit session state to the database."""
    if st.session_state.get('username') and st.session_state.get('session_token'):
        logger.debug(f"Saving session state for user: {st.session_state.username}, token: {st.session_state.session_token}")

        # Create a dictionary of the session state to save (exclude sensitive/large/unpicklable items if necessary)
        state_to_save = {k: v for k, v in st.session_state.items() if k not in ['page_config_set']} # Example exclusion

        try:
            session_blob = pickle.dumps(state_to_save)
        except (pickle.PicklingError, TypeError) as e:
             logger.error(f"Failed to pickle session state for user {st.session_state.username}: {e}. Session not saved.")
             # Optionally remove the problematic key and try again, or just skip saving.
             # For simplicity, we'll skip saving if pickling fails.
             return

        conn = get_db_connection()
        if conn is None: return

        try:
            with conn.cursor() as c:
                # Use ON CONFLICT to update existing session token entry
                query = sql.SQL("""
                    INSERT INTO sessions (session_token, username, session_data, last_accessed)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (session_token) DO UPDATE SET
                        session_data = EXCLUDED.session_data,
                        last_accessed = CURRENT_TIMESTAMP
                """)
                c.execute(query, (st.session_state.session_token, st.session_state.username, psycopg2.Binary(session_blob)))
            conn.commit()
            logger.info(f"Session state saved successfully for user: {st.session_state.username}, token: {st.session_state.session_token}")
        except Exception as e:
            logger.error(f"Error saving session state for user {st.session_state.username}: {str(e)}")
            conn.rollback()
        finally:
            if conn:
                conn.close()

def load_session_state_from_db(session_token: str):
    """Loads session state from the database using a session token."""
    logger.debug(f"Attempting to load session state for token: {session_token}")
    conn = get_db_connection()
    if conn is None: return False

    loaded = False
    try:
        with conn.cursor() as c:
            # Update last_accessed time when loading
            c.execute("""
                UPDATE sessions SET last_accessed = CURRENT_TIMESTAMP
                WHERE session_token = %s
                RETURNING username, session_data
            """, (session_token,))
            result = c.fetchone()

        if result:
            username, session_blob = result
            try:
                loaded_state = pickle.loads(session_blob)
                # Restore the loaded state into st.session_state
                # Be careful not to overwrite essential keys like 'authenticated' prematurely
                for key, value in loaded_state.items():
                     # Avoid overwriting keys managed directly by the login flow during initial load
                    if key not in ['authenticated', 'username', 'user_info', 'session_token', 'page', 'oauth_state', 'login_error', 'signup_error', 'signup_success']:
                        st.session_state[key] = value

                # Explicitly set core authentication state AFTER loading other data
                st.session_state.authenticated = loaded_state.get('authenticated', False)
                st.session_state.username = username
                st.session_state.user_info = loaded_state.get('user_info')
                st.session_state.session_token = session_token
                # Set page based on loaded state, default to Upload if authenticated
                st.session_state.page = loaded_state.get('page', "Upload" if st.session_state.authenticated else "Login")

                logger.info(f"Session restored for user {username} from token {session_token}. Authenticated: {st.session_state.authenticated}, Page: {st.session_state.page}")
                loaded = True
            except (pickle.UnpicklingError, TypeError, EOFError) as e:
                logger.error(f"Failed to unpickle session data for token {session_token}: {e}. Session not restored.")
                # Optionally delete the corrupt session entry
            except Exception as e:
                 logger.error(f"Unexpected error loading session state from DB for token {session_token}: {e}")

        else:
            logger.debug(f"No session found or failed to update for token: {session_token}")
            # Clear potentially invalid token from session_state if load fails
            if 'session_token' in st.session_state and st.session_state.session_token == session_token:
                st.session_state.session_token = None
                st.session_state.authenticated = False # Ensure user is logged out

    except Exception as e:
        logger.error(f"Database error loading session state for token {session_token}: {str(e)}")
    finally:
        if conn:
            conn.commit() # Commit the timestamp update if successful
            conn.close()

    return loaded

def clear_session_from_db(session_token: str):
    """Removes a session record from the database."""
    logger.info(f"Clearing session from DB for token: {session_token}")
    conn = get_db_connection()
    if conn is None: return

    try:
        with conn.cursor() as c:
            c.execute("DELETE FROM sessions WHERE session_token = %s", (session_token,))
        conn.commit()
        logger.info(f"Session cleared successfully for token: {session_token}")
    except Exception as e:
        logger.error(f"Error clearing session from DB for token {session_token}: {str(e)}")
        conn.rollback()
    finally:
        if conn:
            conn.close()

# --- End of Session Management Functions ---


# --- Attempt to restore session from token in query params on initial load ---
# This runs only once per browser session load unless the token changes in URL
if not st.session_state.get('authenticated', False): # Only run if not already authenticated
    query_params = st.query_params.to_dict()
    token_from_url = query_params.get('session_token')
    if token_from_url:
        logger.info(f"Found session_token in query_params: {token_from_url}")
        if load_session_state_from_db(token_from_url):
            # If session loaded successfully, update query params to reflect loaded token
            # Important: Use st.query_params setter
            st.query_params["session_token"] = token_from_url
            # No rerun here, let the rest of the script execute with the loaded state
        else:
            # If token is invalid, remove it from URL to prevent reload loops
             logger.warning(f"Invalid session token found in URL, removing: {token_from_url}")
             st.query_params.clear() # Clear all params, or specifically remove session_token
             # No rerun needed, proceed as unauthenticated
    else:
        logger.debug("No session_token found in query_params on initial load.")
# --- End of Initial Session Restore ---


# --- CSS and UI Functions ---
def load_css(theme: str = "dark") -> None:
    """Loads CSS styles based on the selected theme."""
    # Minified CSS for brevity (consider loading from a file)
    css = """
    body {{ font-family: 'Roboto', sans-serif !important; margin: 0; padding: 0; }}
    body.{theme}-theme {{ display: block !important; }}
    body.dark-theme .stApp {{ background: linear-gradient(to bottom right, #1C2526, #2A3B47) !important; color: #FFFFFF !important; }}
    body.dark-theme .css-1d391kg {{ background-color: #1C2526 !important; color: #FFFFFF !important; }} /* Sidebar */
    body.dark-theme .css-1d391kg .tagline {{ font-size: 16px !important; color: #1E90FF !important; font-style: italic !important; }}
    body.dark-theme h1 {{ color: #1E90FF !important; font-family: 'Roboto', sans-serif !important; }}
    body.dark-theme h2, body.dark-theme h3 {{ color: #FFD700 !important; font-family: 'Roboto', sans-serif !important; }}
    body.dark-theme .stButton > button {{ background-color: #1E90FF !important; color: white !important; border-radius: 5px !important; transition: background-color 0.3s !important; font-family: 'Roboto', sans-serif !important; border: none !important; }}
    body.dark-theme .stButton > button:hover {{ background-color: #FFD700 !important; color: #1C2526 !important; }}
    body.dark-theme .stContainer, body.dark-theme .stExpander {{ background-color: rgba(255, 255, 255, 0.1) !important; border-radius: 10px !important; padding: 15px !important; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important; }}
    body.dark-theme .stTextInput > div > div > input, body.dark-theme .stSelectbox > div > div > div, body.dark-theme .stMultiSelect > div > div > div {{ background-color: #2A3B47 !important; color: #FFFFFF !important; border: 1px solid #1E90FF !important; border-radius: 5px !important; }}
    body.dark-theme .stDataFrame {{ background-color: #2A3B47 !important; border-radius: 10px !important; padding: 10px !important; }}
    body.dark-theme .stProgress > div > div {{ background-color: #1E90FF !important; }}
    body.dark-theme .stAlert {{ background-color: rgba(255, 255, 255, 0.1) !important; color: #FFFFFF !important; border-radius: 5px !important; }}
    body.light-theme .stApp {{ background: linear-gradient(to bottom right, #F0F4F8, #D9E2EC) !important; color: #000000 !important; }}
    body.light-theme .css-1d391kg {{ background-color: #D9E2EC !important; color: #000000 !important; }} /* Sidebar */
    body.light-theme .css-1d391kg .tagline {{ font-size: 16px !important; color: #0066CC !important; font-style: italic !important; }}
    body.light-theme h1 {{ color: #0066CC !important; font-family: 'Roboto', sans-serif !important; }}
    body.light-theme h2, body.light-theme h3 {{ color: #CC9900 !important; font-family: 'Roboto', sans-serif !important; }}
    body.light-theme .stButton > button {{ background-color: #0066CC !important; color: white !important; border-radius: 5px !important; transition: background-color 0.3s !important; font-family: 'Roboto', sans-serif !important; border: none !important; }}
    body.light-theme .stButton > button:hover {{ background-color: #CC9900 !important; color: #FFFFFF !important; }}
    body.light-theme .stContainer, body.light-theme .stExpander {{ background-color: rgba(0, 0, 0, 0.05) !important; border-radius: 10px !important; padding: 15px !important; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important; }}
    body.light-theme .stTextInput > div > div > input, body.light-theme .stSelectbox > div > div > div, body.light-theme .stMultiSelect > div > div > div {{ background-color: #F0F4F8 !important; color: #000000 !important; border: 1px solid #0066CC !important; border-radius: 5px !important; }}
    body.light-theme .stDataFrame {{ background-color: #F0F4F8 !important; border-radius: 10px !important; padding: 10px !important; }}
    body.light-theme .stProgress > div > div {{ background-color: #0066CC !important; }}
    body.light-theme .stAlert {{ background-color: rgba(0, 0, 0, 0.05) !important; color: #000000 !important; border-radius: 5px !important; }}
    .google-login-button {{ display: flex !important; align-items: center !important; justify-content: center !important; background-color: #FFFFFF !important; color: #757575 !important; border: 1px solid #DADCE0 !important; border-radius: 4px !important; padding: 10px 20px !important; font-size: 16px !important; font-family: 'Roboto', sans-serif !important; font-weight: 500 !important; cursor: pointer !important; transition: background-color 0.3s ease, box-shadow 0.3s ease !important; width: 100% !important; box-sizing: border-box !important; margin: 10px auto !important; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1) !important; }}
    .google-login-button:hover {{ background-color: #F8FAFC !important; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important; }}
    .google-login-button img {{ width: 20px !important; height: 20px !important; margin-right: 10px !important; }}
    .google-login-button span {{ color: #757575 !important; font-family: 'Roboto', sans-serif !important; }}
    a.google-login-button {{ text-decoration: none !important; }}
    /* Specific button styling (examples) */
    body.dark-theme .stButton#start_cleaning_button > button {{ background-color: #4CAF50 !important; }}
    body.dark-theme .stButton#start_cleaning_button > button:hover {{ background-color: #45a049 !important; }}
    body.dark-theme .stButton#delete_dataset_button > button {{ background-color: #f44336 !important; }}
    body.dark-theme .stButton#delete_dataset_button > button:hover {{ background-color: #da190b !important; }}
    """
    # Inject CSS using components.html for immediate effect
    components.html(
        f"""
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
        <style>{css}</style>
        <script>
            document.body.className = "{theme}-theme";
            // console.log("Applied body class:", document.body.className); // Optional console log for debugging
        </script>
        """,
        height=0
    )

def render_custom_header(page_title: str):
    """Renders a custom page header."""
    st.markdown(f"<h1 style='margin-top: 20px;'>{page_title}</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid #FFD700;'>", unsafe_allow_html=True)
# --- End of CSS and UI Functions ---


# --- Google OAuth Functions ---
def get_google_auth_url():
    """Generates the Google OAuth authorization URL."""
    base_redirect_uri = GOOGLE_REDIRECT_URI.split('?')[0] # Ensure base URI matches Google Console
    try:
        client = OAuth2Session(
            GOOGLE_CLIENT_ID,
            GOOGLE_CLIENT_SECRET,
            redirect_uri=base_redirect_uri,
            scope=SCOPES)
        auth_url, state = client.create_authorization_url(GOOGLE_AUTH_URL)
        st.session_state['oauth_state'] = state # Store state for verification
        logger.info(f"Generated Google Auth URL with state: {state}")
        return auth_url
    except Exception as e:
        st.error("Failed to generate Google login URL.")
        logger.error(f"Error generating Google Auth URL: {e}")
        return None

def handle_google_callback():
    """Handles the callback from Google OAuth, exchanges code for token, fetches user info."""
    logger.info("--- Entered handle_google_callback ---")
    callback_error = None
    user_info = None

    # Check if state matches (CSRF protection)
    callback_state = st.query_params.get('state', [None])[0]
    saved_state = st.session_state.get('oauth_state')
    logger.debug(f"Callback state: {callback_state}, Saved state: {saved_state}")

    if not saved_state or callback_state != saved_state:
        callback_error = "OAuth state mismatch. Potential CSRF attack or session issue."
        logger.error(callback_error)
        # Clear invalid state from session and URL if possible
        if 'oauth_state' in st.session_state: del st.session_state['oauth_state']
        if 'state' in st.query_params: del st.query_params['state']
        if 'code' in st.query_params: del st.query_params['code'] # Also clear code if state fails
        st.error(callback_error)
        st.session_state['login_error'] = callback_error # Display error on login page
        st.rerun() # Rerun to show the error on the login page
        return None # Explicitly return None

    code = st.query_params.get('code', [None])[0]
    logger.info(f"Attempting to fetch token with code: {'********' if code else 'None'}") # Avoid logging full code

    if not code:
        callback_error = "Authentication failed: No authorization code received from Google."
        logger.error(callback_error)
        st.error(callback_error)
        # Clear params if code is missing
        if 'state' in st.query_params: del st.query_params['state']
        st.session_state['login_error'] = callback_error
        st.rerun()
        return None

    try:
        base_redirect_uri = GOOGLE_REDIRECT_URI.split('?')[0]
        client = OAuth2Session(
            GOOGLE_CLIENT_ID,
            GOOGLE_CLIENT_SECRET,
            redirect_uri=base_redirect_uri,
            state=saved_state) # Use the verified state

        # Exchange code for token
        token = client.fetch_token(GOOGLE_TOKEN_URL, code=code)
        logger.info("Successfully fetched OAuth token.")

        # Fetch user info
        user_info_response = requests.get(
            GOOGLE_USERINFO_URL, headers={'Authorization': f"Bearer {token['access_token']}"}
        )
        user_info_response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        user_info = user_info_response.json()
        logger.info(f"Successfully fetched user info for Google ID: {user_info.get('sub')}")

        # --- IMPORTANT: Clear code and state from query params AFTER successful processing ---
        # This prevents reusing the code if the page is refreshed or rerun
        # Use del on the MutableQueryParams object
        del st.query_params['code']
        del st.query_params['state']
        logger.info("Cleared 'code' and 'state' from query parameters.")
        # --- End of clearing params ---

        # Clear the state from session_state as it's single-use for the callback
        if 'oauth_state' in st.session_state:
             del st.session_state['oauth_state']
             logger.info("Cleared 'oauth_state' from session state.")


    except requests.exceptions.RequestException as e:
        callback_error = f"Network error fetching user info: {str(e)}"
        logger.error(callback_error)
    except Exception as e:
        # This catches errors during fetch_token (like invalid_grant) or JSON parsing
        callback_error = f"Error during Google OAuth token exchange or user info fetch: {str(e)}"
        logger.error(f"Error during Google OAuth token exchange or user info fetch (Code: {'********' if code else 'None'}, State: {saved_state}): {str(e)}", exc_info=True)
        # If the error is likely due to the code (e.g., invalid_grant), clear it
        if 'invalid_grant' in str(e).lower() or 'malformed auth code' in str(e).lower():
             if 'code' in st.query_params: del st.query_params['code']
             if 'state' in st.query_params: del st.query_params['state'] # Clear state too if code fails
             logger.info("Cleared 'code' and 'state' from query parameters due to likely invalid code.")
             if 'oauth_state' in st.session_state: del st.session_state['oauth_state']


    finally:
        logger.info("--- Exiting handle_google_callback ---")

    if callback_error:
        st.error(callback_error) # Show error to user
        st.session_state['login_error'] = callback_error # Store error for login page display
        st.rerun() # Rerun to display error on login page
        return None # Return None on error

    return user_info # Return fetched user info on success
# --- End of Google OAuth Functions ---


# --- Authentication and Page Routing Logic ---

# Handle Login Page
if not st.session_state.get('authenticated'):
    st.session_state.page = "Login" # Force page to Login if not authenticated

    # Check if Google OAuth callback is happening (code and state in URL)
    # This check needs to happen *before* rendering the login page content
    # to allow the callback handler to process and potentially authenticate.
    if 'code' in st.query_params and 'state' in st.query_params:
        logger.info("Detected 'code' and 'state' in query_params, attempting Google callback.")
        google_user_info = handle_google_callback()

        if google_user_info:
            google_id = google_user_info['sub']
            user = get_user_by_google_id(google_id) # Check if Google user exists in DB

            if user: # Existing user linked to Google ID
                username = user['username']
                logger.info(f"Google login successful for existing user: {username}")
            else: # New user via Google login
                username = google_user_info['email'].split('@')[0] # Default username
                email = google_user_info['email']
                name = google_user_info['name']
                profile_picture = google_user_info.get('picture')
                # Attempt to add the new user
                if add_user(username, email, name, google_id=google_id, profile_picture=profile_picture):
                     logger.info(f"New user created via Google login: {username}")
                else:
                     # Handle potential error during user creation (e.g., username collision, DB error)
                     st.error(st.session_state.get('signup_error', "Failed to register Google user. Please try again or contact support."))
                     st.session_state.page = "Login" # Stay on login page
                     st.rerun() # Rerun to show error


            # --- Set authenticated state ---
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.user_info = google_user_info # Store Google info
            st.session_state.page = "Upload" # Redirect to main app page

            # --- Generate and store session token ---
            st.session_state.session_token = str(uuid.uuid4())
            logger.info(f"Generated new session token for user {username}: {st.session_state.session_token}")
            save_session_state_to_db() # Save the initial authenticated state with token

            # --- Update URL with session token (replaces OAuth params) ---
            # We already cleared code/state, now set the session token
            st.query_params.clear() # Ensure clean slate
            st.query_params["session_token"] = st.session_state.session_token
            logger.info("Updated query params with session_token after Google login.")

            st.rerun() # Rerun to load the main app page ("Upload")

        else:
            # handle_google_callback already showed error and triggered rerun
            # Execution should stop here if callback failed
            logger.warning("Google callback handled but resulted in no user info.")
            # No further action needed here, rerun was called in handler

    # --- Render Login/Sign Up Page Content ---
    # This part runs if not authenticated AND not currently handling a successful OAuth callback
    load_css(st.session_state.theme) # Load CSS for login/signup page

    if st.session_state.page == "Login":
        st.markdown(f"""<div class="login-card" style="background: {'#2A3B47' if st.session_state.theme == 'dark' else '#FFFFFF'}; border-radius: 15px; padding: 30px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); max-width: 400px; margin: 100px auto 0;">""", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center; margin-bottom: 20px; font-size: 24px; color: {'#1E90FF' if st.session_state.theme == 'dark' else '#0066CC'};'>Welcome to Data Toy AI</h1>", unsafe_allow_html=True)

        # Display login errors if any
        if st.session_state.get('login_error'):
            st.error(st.session_state.login_error)
            st.session_state.login_error = None # Clear error after displaying

        # Display signup success message if redirected from signup
        if st.session_state.get('signup_success'):
             st.success(st.session_state.signup_success)
             st.session_state.signup_success = None # Clear message

        # --- Username/Password Form ---
        login_form = st.form("login_form")
        with login_form:
            username_input = st.text_input("Username", key="login_username")
            password_input = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login")

            if submitted:
                user_data = verify_user(username_input, password_input)
                if user_data:
                    st.session_state.authenticated = True
                    st.session_state.username = user_data['username']
                    st.session_state.user_info = user_data # Store basic user info
                    st.session_state.page = "Upload"
                    # Generate and store session token
                    st.session_state.session_token = str(uuid.uuid4())
                    logger.info(f"Generated new session token for user {st.session_state.username}: {st.session_state.session_token}")
                    save_session_state_to_db() # Save initial authenticated state
                    # Update URL
                    st.query_params.clear()
                    st.query_params["session_token"] = st.session_state.session_token
                    st.rerun()
                else:
                    # Error message is set within verify_user or defaults if user not found
                    st.session_state.login_error = st.session_state.get('login_error', "Incorrect username or password.")
                    st.rerun() # Rerun to display the error message


        # --- Google Login Button ---
        auth_url = get_google_auth_url()
        if auth_url:
             st.markdown(f"""<a href="{auth_url}" target="_self" class="google-login-button" style="text-decoration: none; margin-top: 15px;"> <img src="https://developers.google.com/identity/images/g-logo.png" alt="Google Icon"/> <span>Sign in with Google</span> </a>""", unsafe_allow_html=True)
        else:
             st.error("Google Login is currently unavailable.")


        # --- Link to Sign Up ---
        st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)
        if st.button("Don't have an account? Sign Up", key="goto_signup"):
            st.session_state.page = "Sign Up"
            st.session_state.login_error = None # Clear login errors
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True) # Close login-card div


    elif st.session_state.page == "Sign Up":
        st.markdown(f"""<div class="login-card" style="background: {'#2A3B47' if st.session_state.theme == 'dark' else '#FFFFFF'}; border-radius: 15px; padding: 30px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); max-width: 400px; margin: 100px auto 0;">""", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center; margin-bottom: 20px; font-size: 24px; color: {'#1E90FF' if st.session_state.theme == 'dark' else '#0066CC'};'>Sign Up for Data Toy AI</h1>", unsafe_allow_html=True)

        # Display signup errors if any
        if st.session_state.get('signup_error'):
            st.error(st.session_state.signup_error)
            st.session_state.signup_error = None # Clear error after displaying

        signup_form = st.form("signup_form")
        with signup_form:
            new_username = st.text_input("Username*", key="signup_username")
            new_email = st.text_input("Email*", key="signup_email")
            new_name = st.text_input("Name", key="signup_name")
            new_password = st.text_input("Password*", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password*", type="password", key="signup_confirm_password")
            submitted = st.form_submit_button("Register")

            if submitted:
                # Basic Validation
                if not new_username or not new_email or not new_password or not confirm_password:
                     st.session_state['signup_error'] = "Please fill in all required fields (*)."
                elif new_password != confirm_password:
                     st.session_state['signup_error'] = "Passwords do not match."
                elif '@' not in new_email or '.' not in new_email: # Basic email format check
                     st.session_state['signup_error'] = "Please enter a valid email address."
                else:
                    if add_user(new_username, new_email, new_name or '', new_password): # Pass empty string if name is optional
                        st.session_state.page = "Login"
                        st.session_state.signup_success = "Registration successful! Please log in."
                        st.session_state.signup_error = None # Clear any previous error
                        st.rerun()
                    else:
                        # Error message should be set in add_user if DB error occurs
                        st.session_state.signup_error = st.session_state.get('signup_error', "Registration failed. The username or email might already exist.")
                        st.rerun() # Rerun to display the error

        # --- Link back to Login ---
        st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)
        if st.button("Already have an account? Login", key="goto_login"):
            st.session_state.page = "Login"
            st.session_state.signup_error = None # Clear signup errors
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True) # Close login-card div

# --- Main Application Logic (Runs only if authenticated) ---
if st.session_state.get('authenticated'):
    # Load CSS with selected theme
    load_css(st.session_state.theme)

    # --- Sidebar Setup ---
    def setup_sidebar(logo_path: str = "images/datatoy_logo.png") -> Optional[str]: # Make sure logo path is correct
        """Sets up the sidebar navigation and elements."""
        # Logo
        try:
            # Check if the file exists before trying to display it
            if os.path.exists(logo_path):
                st.sidebar.image(logo_path, use_container_width=True)
            else:
                 logger.warning(f"Sidebar logo not found at path: {logo_path}")
                 st.sidebar.markdown("**Data Toy AI**", unsafe_allow_html=True) # Fallback text
        except Exception as e:
             logger.error(f"Error loading sidebar logo '{logo_path}': {e}")
             st.sidebar.markdown("**Data Toy AI**", unsafe_allow_html=True)

        # Welcome Message & Profile Picture
        user_display_name = st.session_state.username # Default
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

        # Page Navigation (Radio Buttons)
        pages = ["Upload", "Clean", "Insights", "Visualize", "Predictive", "Share"]
        try:
            current_page_index = pages.index(st.session_state.page)
        except ValueError:
            current_page_index = 0 # Default to Upload if current page isn't in list
            st.session_state.page = pages[0]

        selected_page = st.sidebar.radio(
            "Go to", pages, index=current_page_index, key="sidebar_nav"
        )

        # Update page state and save session if navigation changes
        if selected_page != st.session_state.page:
            st.session_state.page = selected_page
            save_session_state_to_db() # Save state on page change
            st.rerun() # Rerun to render the new page

        # Theme Toggle
        st.sidebar.subheader("Theme")
        theme_options = ["Dark", "Light"]
        current_theme_index = 0 if st.session_state.theme == "dark" else 1
        theme_choice = st.sidebar.selectbox("Select Theme", theme_options, index=current_theme_index, key="theme_select")
        new_theme = theme_choice.lower()
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            save_session_state_to_db()
            st.rerun()

        # Progress Tracker
        st.sidebar.subheader("Your Progress")
        progress_text = ""
        for step, status in st.session_state.progress.items():
            emoji = "✅" if status == "Done" else "🟡" if status == "In Progress" else "⬜"
            progress_text += f"{emoji} {step}: {status}\n"
        st.sidebar.markdown(f"```\n{progress_text}\n```") # Use code block for alignment

        # AI Assistant Warning
        if not AI_AVAILABLE:
            st.sidebar.error("⚠️ AI features disabled (OpenAI key missing/invalid).")

        # AI Chat Assistant
        st.sidebar.subheader("AI Data Assistant")
        # Use an expander for the chat history to save space
        with st.sidebar.expander("Chat History", expanded=False):
             if not st.session_state.chat_history:
                 st.write("No chat history yet.")
             else:
                for message in st.session_state.chat_history:
                    role = message.get("role", "unknown")
                    content = message.get("content", "")
                    with st.chat_message(role):
                        st.write(content)

        # Chat Input
        chat_input = st.sidebar.chat_input("Ask Data Toy about your data...")
        if chat_input:
            # Determine which DataFrame to use for context
            df_context = st.session_state.get('cleaned_df') if st.session_state.get('cleaned_df') is not None else st.session_state.get('df')

            if df_context is not None and AI_AVAILABLE:
                st.session_state.chat_history.append({"role": "user", "content": chat_input})
                with st.spinner("AI Assistant is thinking..."):
                    try:
                        response = chat_with_gpt(df_context, chat_input, max_tokens=150) # Use function from data_utils
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as chat_e:
                         logger.error(f"Error calling chat_with_gpt: {chat_e}")
                         st.session_state.chat_history.append({"role": "assistant", "content": "Sorry, I encountered an error trying to respond."})

                save_session_state_to_db() # Save state after chat interaction
                st.rerun()
            elif not AI_AVAILABLE:
                 st.sidebar.warning("AI Assistant is disabled. Please configure OpenAI API key.")
            else:
                st.sidebar.warning("Please upload or clean a dataset first to use the AI assistant.")

        # Other Sidebar Links
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Feedback & Community**")
        st.sidebar.markdown("- [Share Feedback](https://docs.google.com/forms/d/e/1FAIpQLScpUFM0Y5_i5LJDM-HZEZEtOHbLHy4Vp-ek_-819MRZo7Q9rQ/viewform?usp=dialog)") # Replace with your actual links
        st.sidebar.markdown("- [Join Discord](https://discord.gg/your-invite-link)")
        st.sidebar.markdown("**Support & Upgrade**")
        st.sidebar.markdown("- [Help Documentation](https://your-docs-link.com)")
        st.sidebar.markdown("- [Upgrade to Premium ($5/mo)](https://stripe.com/your-checkout-link)")

        # Dev Mode Indicator (Optional)
        is_dev_mode = os.getenv("DEV_MODE") == "true"
        if is_dev_mode:
            st.sidebar.info("DEV_MODE Active")

        # Logout Button
        st.sidebar.markdown("---")
        if st.sidebar.button("Logout", key="logout_button"):
            logout_token = st.session_state.get('session_token')
            # Clear local Streamlit session state first
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            # Explicitly set necessary keys for redirect
            st.session_state.authenticated = False
            st.session_state.page = "Login"
            # Clear session from DB if token exists
            if logout_token:
                clear_session_from_db(logout_token)
            # Clear query parameters
            st.query_params.clear()
            logger.info(f"User logged out. Session token {logout_token} cleared.")
            st.rerun()

        return selected_page # Return the currently selected page
    # --- End of Sidebar Setup ---


    # --- Main Page Rendering Logic ---
    def render_main_content():
        """Renders the content for the selected page."""
        current_page = setup_sidebar() # Setup sidebar and get current page selection

        if not current_page:
            st.error("Page selection failed.")
            logger.error("setup_sidebar did not return a valid page.")
            return

        page_titles = {
            "Upload": "Upload Your Dataset", "Clean": "Clean Your Dataset",
            "Insights": "Insights Dashboard", "Visualize": "Visualize Your Dataset",
            "Predictive": "Predictive Analytics", "Share": "Share Your Work"
        }
        render_custom_header(page_titles.get(current_page, "Data Toy AI"))

        try:
            # Get current DataFrame context (cleaned first, then original)
            df_context = st.session_state.get('cleaned_df') if st.session_state.get('cleaned_df') is not None else st.session_state.get('df')

            # Page specific rendering
            if current_page == "Upload":
                render_upload_page() # Assumes this function exists in ui.py
            elif current_page == "Clean":
                 if df_context is None:
                     st.warning("Please upload a dataset on the 'Upload' page first.")
                 else:
                    render_clean_page() # Assumes this function exists in ui.py
            elif current_page == "Insights":
                 if df_context is None:
                     st.warning("Please upload or clean a dataset first.")
                 else:
                    render_insights_page() # Assumes this function exists in ui.py
            elif current_page == "Visualize":
                 if df_context is None:
                     st.warning("Please upload or clean a dataset first.")
                 else:
                    render_visualization_page(df_context) # Assumes this function exists in visualizations.py
            elif current_page == "Predictive":
                 if df_context is None:
                     st.warning("Please upload or clean a dataset first.")
                 else:
                    render_predictive_page(df_context) # Assumes this function exists in ui.py
            elif current_page == "Share":
                st.info("Sharing and collaboration features are under development. Stay tuned!")
                st.session_state.progress["Share"] = "Done" # Mark as done for now

        except Exception as e:
            st.error(f"An error occurred while rendering the '{current_page}' page: {str(e)}")
            logger.error(f"Error rendering page '{current_page}': {str(e)}", exc_info=True)
            if current_page in st.session_state.progress:
                 st.session_state.progress[current_page] = "Failed"

        # Save session state to DB after every interaction/page render
        # This ensures the latest state is persisted frequently
        save_session_state_to_db()

    # --- Run the main content rendering function ---
    render_main_content()

# --- End of Authenticated App Logic ---
