import pandas as pd
import numpy as np
import openai
from openai import OpenAI
import streamlit as st
import re
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cluster import KMeans
import requests
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import joblib
from sklearn.datasets import make_classification, make_regression
import logging
from logging.handlers import RotatingFileHandler
import os
import httpx
from typing import Dict, List, Tuple, Optional, Union
from ratelimit import limits, sleep_and_retry
from cryptography.fernet import Fernet
import sqlite3
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = RotatingFileHandler('data_utils.log', maxBytes=5*1024*1024, backupCount=3)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# Fix 22: Handle OpenAI authentication errors specifically
api_key = None
try:
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found.")
    logger.info("Successfully loaded OPENAI_API_KEY")
except ValueError as e:
    logger.error(f"Failed to load OpenAI API key: {str(e)}")
    st.error("OpenAI API key is missing. Configure it in secrets or environment.")
except Exception as e:
    logger.error(f"Unexpected error loading OpenAI API key: {str(e)}")
    st.error("Error loading OpenAI API key.")

client = None
if api_key:
    try:
        http_client = httpx.Client(proxies=None)
        client = OpenAI(api_key=api_key, http_client=http_client)
        logger.info("OpenAI client initialized with version: %s", openai.__version__)
    except openai.AuthenticationError as e:
        # Fix 22: Specific handling for authentication error
        logger.error(f"OpenAI authentication failed: {str(e)}")
        st.error("Invalid OpenAI API key. AI features disabled.")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {str(e)}")
        st.error("Failed to initialize OpenAI client.")

AI_AVAILABLE = client is not None

# Fix 24: Use persistent encryption key from st.secrets
ENCRYPTION_KEY = st.secrets.get("ENCRYPTION_KEY", Fernet.generate_key())  # Fallback to generate if not set
cipher = Fernet(ENCRYPTION_KEY)

def encrypt_dataframe(df: pd.DataFrame) -> bytes:
    try:
        df_bytes = df.to_pickle(None)
        encrypted_data = cipher.encrypt(df_bytes)
        return encrypted_data
    except Exception as e:
        logger.error(f"Error encrypting DataFrame: {str(e)}")
        return None

def decrypt_dataframe(encrypted_data: bytes) -> pd.DataFrame:
    try:
        df_bytes = cipher.decrypt(encrypted_data)
        df = pd.read_pickle(df_bytes)
        return df
    except Exception as e:
        logger.error(f"Error decrypting DataFrame: {str(e)}")
        return None

def init_analytics_db():
    conn = sqlite3.connect('datatoy_analytics.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS analytics 
                 (username TEXT, action TEXT, timestamp TEXT)''')
    conn.commit()
    conn.close()

def log_action(username: str, action: str):
    init_analytics_db()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn = sqlite3.connect('datatoy_analytics.db')
    c = conn.cursor()
    c.execute("INSERT INTO analytics (username, action, timestamp) VALUES (?, ?, ?)", 
              (username, action, timestamp))
    conn.commit()
    conn.close()

CALLS_PER_MINUTE = 10
@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=60)
def rate_limited_api_call(func, *args, **kwargs):
    return func(*args, **kwargs)

def detect_outliers(df: pd.DataFrame, col: str, method: str = "iqr", contamination: float = 0.1) -> Tuple[bool, float, float]:
    try:
        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            return len(outliers) > 0, lower_bound, upper_bound
        elif method == "isolation_forest":
            data = df[[col]].dropna()
            if data.empty:
                return False, 0, 0
            model = IsolationForest(contamination=contamination, random_state=42)
            predictions = model.fit_predict(data)
            outliers = data[predictions == -1]
            return len(outliers) > 0, data[col].min(), data[col].max()
        else:
            raise ValueError(f"Unsupported method: {method}")
    except Exception as e:
        logger.error(f"Error in detect_outliers for column {col}: {str(e)}")
        st.error(f"Failed to detect outliers in column {col}: {str(e)}")
        return False, 0, 0

def detect_anomalies(df: pd.DataFrame, cols: List[str], contamination: float = 0.1) -> Dict[str, Dict]:
    if isinstance(df, bytes):
        df = decrypt_dataframe(df)

    anomalies = {}
    try:
        for col in cols:
            data = df[[col]].dropna()
            if not data.empty:
                model = IsolationForest(contamination=contamination, random_state=42)
                predictions = model.fit_predict(data)
                anomaly_indices = data[predictions == -1].index
                anomalies[col] = df.loc[anomaly_indices, col].to_dict()
            else:
                logger.warning(f"No data available for anomaly detection in column {col}")
    except Exception as e:
        logger.error(f"Error in detect_anomalies: {str(e)}")
        st.error(f"Failed to detect anomalies: {str(e)}")
    return anomalies

@st.cache_data
def analyze_dataset(df: pd.DataFrame) -> Dict[str, Union[int, List[str], bool]]:
    if isinstance(df, bytes):
        df = decrypt_dataframe(df)

    try:
        analysis = {
            "has_question_marks": '?' in df.values,
            "special_char_cols": [col for col in df.columns if any(c in col for c in "#@$%^&* ()")],
            "empty_rows": df.isna().all(axis=1).sum(),
            "missing_cols": df.columns[df.isna().any()].tolist(),
            "numeric_cols": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            "cat_cols": df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "duplicates": df.duplicated().sum(),
            "time_cols": [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        }
        analysis["type_issues"] = {}
        for col in df.columns:
            col_types = df[col].apply(type).nunique()
            if col_types > 1:
                analysis["type_issues"][col] = {
                    "mixed_types": True,
                    "suggested_type": df[col].dtype.name if pd.api.types.is_numeric_dtype(df[col]) else "string"
                }
            elif df[col].dtype == 'object':
                try:
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    if numeric_series.notna().mean() > 0.9:
                        analysis["type_issues"][col] = {
                            "mixed_types": False,
                            "suggested_type": "numeric"
                        }
                except:
                    pass
        return analysis
    except Exception as e:
        logger.error(f"Error in analyze_dataset: {str(e)}")
        st.error(f"Failed to analyze dataset: {str(e)}")
        return {}

def calculate_health_score(df: pd.DataFrame) -> float:
    if isinstance(df, bytes):
        df = decrypt_dataframe(df)

    try:
        analysis = analyze_dataset(df)
        score = 100.0
        weights = {
            "has_question_marks": 10,
            "special_char_cols": 5,
            "empty_rows": 2,
            "missing_cols": 5,
            "duplicates": 2,
            "type_issues": 3
        }
        if analysis.get("has_question_marks"):
            score -= weights["has_question_marks"]
        if analysis.get("special_char_cols"):
            score -= weights["special_char_cols"] * len(analysis["special_char_cols"])
        if analysis.get("empty_rows"):
            score -= min(20, analysis["empty_rows"] * weights["empty_rows"])
        if analysis.get("missing_cols"):
            score -= min(30, len(analysis["missing_cols"]) * weights["missing_cols"])
        if analysis.get("duplicates"):
            score -= min(20, analysis["duplicates"] * weights["duplicates"])
        if analysis.get("type_issues"):
            score -= len(analysis["type_issues"]) * weights["type_issues"]
        return max(0, score)
    except Exception as e:
        logger.error(f"Error in calculate_health_score: {str(e)}")
        st.error(f"Failed to calculate health score: {str(e)}")
        return 0

@st.cache_data
def get_cleaning_suggestions(df: pd.DataFrame) -> List[Tuple[str, str]]:
    if isinstance(df, bytes):
        df = decrypt_dataframe(df)

    username = st.session_state.get('username', 'anonymous')
    log_action(username, "Generated cleaning suggestions")

    if not client:
        return [("Manual cleaning required", "Configure a valid OpenAI API key.")]

    try:
        analysis = analyze_dataset(df)
        # Fix 26: Use df.describe() for large datasets to reduce prompt size
        preview = df.describe().to_string() if len(df) > 1000 else df.head(10).to_string()
        prompt = f"""
        You are an expert data analyst. Based on this dataset analysis, provide specific, actionable cleaning suggestions with brief explanations:
        - Dataset preview: {preview}
        - Analysis: {analysis}
        Suggest only applicable operations with specific wording and explanations:
        1. "Replace '?' with NaN" if '?' exists - "Converts ambiguous markers to standard missing values."
        2. "Handle special characters in columns: [list]" if special chars exist - "Improves column name usability."
        3. "Remove fully empty rows" if empty rows exist - "Eliminates useless data points."
        4. "Fill missing values in [col] with [mean/median/mode]" for each column with missing values - "Restores data completeness."
        5. "Encode categorical column: [col]" for each categorical column - "Prepares for numerical analysis."
        6. "Remove duplicate rows" if duplicates exist - "Ensures data uniqueness."
        7. "Handle outliers in [col]" for each numeric column with outliers - "Reduces data skew."
        8. "Interpolate time series in [col]" if time series columns exist - "Fills gaps in temporal data."
        Format each suggestion as: "Suggestion - Explanation"
        """
        response = rate_limited_api_call(
            client.chat.completions.create,
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700
        )
        suggestions = response.choices[0].message.content.strip().split("\n")
        return [(s.split(" - ")[0].strip("1234567890. "), s.split(" - ")[1] if " - " in s else "No explanation provided") 
                for s in suggestions if s.strip()]
    except Exception as e:
        logger.error(f"Error in get_cleaning_suggestions: {str(e)}")
        st.error(f"Failed to generate AI cleaning suggestions: {str(e)}")
        return [("Error: Failed to generate suggestions", str(e))]

@st.cache_data
def get_insights(df: pd.DataFrame) -> List[str]:
    if isinstance(df, bytes):
        df = decrypt_dataframe(df)

    username = st.session_state.get('username', 'anonymous')
    log_action(username, "Generated insights")

    if not client:
        return ["Configure a valid OpenAI API key for insights."]

    try:
        # Fix 26: Use df.describe() for large datasets
        preview = df.describe().to_string() if len(df) > 1000 else df.head(10).to_string()
        prompt = """
        You are an AI data analyst. Analyze this dataset and provide 3-5 human-readable insights in plain English:
        - Dataset preview: {preview}
        - Analysis: {analysis}
        Examples of insights:
        - "Column X has a strong correlation with Column Y."
        - "Sales increased by 20% in Q3, driven by Region A."
        - "30% of the data in Column Z is missing."
        """
        formatted_prompt = prompt.format(
            preview=preview,
            analysis=analyze_dataset(df)
        )
        response = rate_limited_api_call(
            client.chat.completions.create,
            model="gpt-4o",
            messages=[{"role": "user", "content": formatted_prompt}],
            max_tokens=300
        )
        return response.choices[0].message.content.strip().split("\n")
    except Exception as e:
        logger.error(f"Error in get_insights: {str(e)}")
        st.error(f"Failed to generate insights: {str(e)}")
        return [f"Error: Failed to generate insights - {str(e)}"]

def suggest_visualization(df: pd.DataFrame) -> Tuple[str, str]:
    if isinstance(df, bytes):
        df = decrypt_dataframe(df)

    try:
        analysis = analyze_dataset(df)
        if analysis["time_cols"]:
            return "Line", "Visualize trends over time."
        elif len(analysis["numeric_cols"]) >= 2:
            return "Scatter", "Explore relationships between numerical variables."
        elif len(analysis["cat_cols"]) > 0 and len(analysis["numeric_cols"]) > 0:
            return "Bar", "Compare categories."
        else:
            return "Histogram", "Understand distribution."
    except Exception as e:
        logger.error(f"Error in suggest_visualization: {str(e)}")
        st.error(f"Failed to suggest visualization: {str(e)}")
        return "Bar", "Default suggestion due to error."

@st.cache_data
def suggest_feature_engineering(df: pd.DataFrame) -> List[Tuple[str, str]]:
    if isinstance(df, bytes):
        df = decrypt_dataframe(df)

    username = st.session_state.get('username', 'anonymous')
    log_action(username, "Suggested feature engineering")

    if not client:
        return [("Manual feature engineering required", "Configure a valid OpenAI API key.")]

    try:
        analysis = analyze_dataset(df)
        # Fix 26: Use df.describe() for large datasets
        preview = df.describe().to_string() if len(df) > 1000 else df.head(10).to_string()
        prompt = """
        You are an expert data scientist. Suggest 2-5 new features to engineer with brief explanations:
        - Dataset preview: {preview}
        - Analysis: {analysis}
        Examples:
        - "Create feature: X/Y ratio - Captures relative magnitude."
        - "Create feature: Log of Z - Reduces skewness."
        - "Create feature: X * Y interaction - Captures combined effect."
        Format each suggestion as: "Suggestion - Explanation"
        """
        formatted_prompt = prompt.format(
            preview=preview,
            analysis=analysis
        )
        response = rate_limited_api_call(
            client.chat.completions.create,
            model="gpt-4o",
            messages=[{"role": "user", "content": formatted_prompt}],
            max_tokens=300
        )
        suggestions = response.choices[0].message.content.strip().split("\n")
        return [(s.split(" - ")[0].strip(), s.split(" - ")[1] if " - " in s else "No explanation provided") 
                for s in suggestions if s.strip()]
    except Exception as e:
        logger.error(f"Error in suggest_feature_engineering: {str(e)}")
        st.error(f"Failed to suggest features: {str(e)}")
        return [("Error: Failed to suggest features", str(e))]

def extract_column(suggestion: str) -> Optional[str]:
    # Fix 23: Safely handle regex failure
    try:
        match = re.search(r"in\s+['\"]?(.*?)['\"]?\s*(?:with|$)", suggestion)
        return match.group(1) if match else None
    except (AttributeError, IndexError) as e:
        logger.error(f"Regex failed in extract_column: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in extract_column: {str(e)}")
        return None

def enrich_with_geolocation(df: pd.DataFrame, address_col: str, api_key: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    if isinstance(df, bytes):
        df = decrypt_dataframe(df)

    username = st.session_state.get('username', 'anonymous')
    log_action(username, f"Enriched data with geolocation for column {address_col}")

    if not api_key:
        return df, "No Google API key provided."
    try:
        df[f"{address_col}_lat"] = np.nan
        df[f"{address_col}_lon"] = np.nan
        for i, address in enumerate(df[address_col].dropna()):
            url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}"
            response = requests.get(url).json()
            if response["status"] == "OK":
                lat = response["results"][0]["geometry"]["location"]["lat"]
                lon = response["results"][0]["geometry"]["location"]["lng"]
                df.loc[df[address_col] == address, f"{address_col}_lat"] = lat
                df.loc[df[address_col] == address, f"{address_col}_lon"] = lon
            else:
                logger.warning(f"Geolocation failed for address: {address}")
        return df, f"Enriched {address_col} with coordinates."
    except Exception as e:
        logger.error(f"Error in enrich_with_geolocation: {str(e)}")
        st.error(f"Geolocation enrichment failed: {str(e)}")
        return df, f"Geolocation enrichment failed: {str(e)}"

def interpolate_time_series(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if isinstance(df, bytes):
        df = decrypt_dataframe(df)

    try:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].interpolate(method='linear')
        return df
    except Exception as e:
        logger.error(f"Error in interpolate_time_series: {str(e)}")
        st.error(f"Failed to interpolate time series in column {col}: {str(e)}")
        return df

def analyze_time_series(df: pd.DataFrame, col: str, period: int = 12) -> Dict[str, pd.Series]:
    if isinstance(df, bytes):
        df = decrypt_dataframe(df)

    try:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        decomposition = seasonal_decompose(df[col].dropna(), model='additive', period=period)
        return {
            "trend": decomposition.trend,
            "seasonal": decomposition.seasonal,
            "residual": decomposition.resid
        }
    except Exception as e:
        logger.error(f"Error in analyze_time_series: {str(e)}")
        st.error(f"Failed to analyze time series in column {col}: {str(e)}")
        return {}

def forecast_time_series(df: pd.DataFrame, col: str, periods: int = 5, time_col: Optional[str] = None, freq: str = 'D') -> pd.DataFrame:
    if isinstance(df, bytes):
        df = decrypt_dataframe(df)

    username = st.session_state.get('username', 'anonymous')
    log_action(username, f"Performed time series forecast for column {col}")

    try:
        if time_col and time_col in df.columns:
            df = df.set_index(time_col)
        elif not pd.api.types.is_datetime64_any_dtype(df.index):
            raise ValueError("DataFrame index must be datetime or a time_col must be provided.")
        model = ARIMA(df[col].dropna(), order=(1, 1, 1))
        fitted = model.fit()
        forecast = fitted.forecast(steps=periods)
        forecast_df = pd.DataFrame({col: forecast}, index=pd.date_range(start=df.index[-1], periods=periods+1, freq=freq)[1:])
        return forecast_df
    except Exception as e:
        logger.error(f"Error in forecast_time_series: {str(e)}")
        st.error(f"Failed to forecast time series in column {col}: {str(e)}")
        return pd.DataFrame()

def generate_synthetic_data(df: pd.DataFrame, task_type: str = "classification") -> pd.DataFrame:
    if isinstance(df, bytes):
        df = decrypt_dataframe(df)

    username = st.session_state.get('username', 'anonymous')
    log_action(username, f"Generated synthetic data for {task_type} task")

    try:
        n_samples = len(df)
        n_features = len(df.columns) - 1
        if task_type == "classification":
            X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=max(2, n_features-2), random_state=42)
        else:
            X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=0.1, random_state=42)
        
        synthetic_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
        synthetic_df["target"] = y
        return synthetic_df
    except Exception as e:
        logger.error(f"Error in generate_synthetic_data: {str(e)}")
        st.error(f"Failed to generate synthetic data: {str(e)}")
        return pd.DataFrame()

def auto_feature_engineering(df: pd.DataFrame, feature_cols: List[str], degree: int = 2) -> pd.DataFrame:
    if isinstance(df, bytes):
        df = decrypt_dataframe(df)

    try:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X = df[feature_cols].fillna(0)
        poly_features = poly.fit_transform(X)
        poly_feature_names = poly.get_feature_names_out(feature_cols)
        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
        return pd.concat([df.drop(columns=feature_cols), poly_df], axis=1)
    except Exception as e:
        logger.error(f"Error in auto_feature_engineering: {str(e)}")
        st.error(f"Failed to perform feature engineering: {str(e)}")
        return df

def train_ml_model(df: pd.DataFrame, target_col: str, feature_cols: List[str], task_type: str = "classification", model_type: str = "RandomForest") -> Tuple[Optional[object], float, Optional[object], Optional[np.ndarray], Optional[pd.DataFrame]]:
    if isinstance(df, bytes):
        df = decrypt_dataframe(df)

    username = st.session_state.get('username', 'anonymous')
    log_action(username, f"Trained {model_type} model for {task_type} task")

    try:
        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if model_type == "RandomForest":
            if task_type == "classification":
                model = RandomForestClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10]
                }
            else:
                model = RandomForestRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10]
                }
        elif model_type == "XGBoost":
            if task_type == "classification":
                model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.3]
                }
            else:
                model = xgb.XGBRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.3]
                }
        elif model_type == "LightGBM":
            if task_type == "classification":
                model = lgb.LGBMClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.3]
                }
            else:
                model = lgb.LGBMRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.3]
                }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy' if task_type == "classification" else 'r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        score = best_model.score(X_test, y_test)
        
        explainer = None
        shap_values = None
        try:
            import shap
            if model_type == "RandomForest":
                explainer = shap.TreeExplainer(best_model)
            else:
                explainer = shap.KernelExplainer(best_model.predict, X_test)
            shap_values = explainer.shap_values(X_test)
        except ImportError:
            logger.warning("SHAP library not installed.")
        except Exception as e:
            logger.warning(f"SHAP computation failed: {str(e)}.")
        
        joblib.dump(best_model, "model.pkl")
        return best_model, score, explainer, shap_values, X_test
    except Exception as e:
        logger.error(f"Error in train_ml_model: {str(e)}")
        st.error(f"Failed to train ML model: {str(e)}")
        return None, 0, None, None, None

def perform_clustering(df: pd.DataFrame, feature_cols: List[str], n_clusters: int = 3) -> np.ndarray:
    if isinstance(df, bytes):
        df = decrypt_dataframe(df)

    username = st.session_state.get('username', 'anonymous')
    log_action(username, f"Performed clustering with {n_clusters} clusters")

    try:
        X = df[feature_cols].fillna(0)
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X)
        return labels
    except Exception as e:
        logger.error(f"Error in perform_clustering: {str(e)}")
        st.error(f"Failed to perform clustering: {str(e)}")
        return np.zeros(len(df))

def generate_ml_app(df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> str:
    if isinstance(df, bytes):
        df = decrypt_dataframe(df)

    try:
        app_code = f"""
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")

st.title("Predict {target_col}")

inputs = {{}}
"""
        for col in feature_cols:
            app_code += f"inputs['{col}'] = st.number_input('{col}', value=0.0)\n"
        
        app_code += f"""
df = pd.DataFrame([inputs])
prediction = model.predict(df)[0]
st.write(f"Predicted {target_col}: {{prediction}}")
"""
        with open("predictor_app.py", "w") as f:
            f.write(app_code)
        return "predictor_app.py generated! Run it with 'streamlit run predictor_app.py'"
    except Exception as e:
        logger.error(f"Error in generate_ml_app: {str(e)}")
        st.error(f"Failed to generate ML app: {str(e)}")
        return f"Error: Failed to generate ML app - {str(e)}"

def chat_with_gpt(df: pd.DataFrame, message: str, max_tokens: int = 100) -> str:
    if isinstance(df, bytes):
        df = decrypt_dataframe(df)

    username = st.session_state.get('username', 'anonymous')
    log_action(username, "Used AI chat assistant")

    if not client:
        return "Configure a valid OpenAI API key to enable AI chat."
    
    identity_keywords = ["who are you", "what are you", "who created you", "what's your name"]
    if any(keyword in message.lower() for keyword in identity_keywords):
        return "I’m your data assistant, built for data analysis."
    
    try:
        analysis = analyze_dataset(df)
        # Fix 26: Use df.describe() for large datasets
        preview = df.describe().to_string() if len(df) > 1000 else df.head(10).to_string()
        prompt = """
        You are customer's data assistant, an AI built for data analysis. Respond to this user message based on the dataset analysis:
        - Analysis: {analysis}
        - Dataset preview: {preview}
        User message: "{message}"
        Provide a helpful response, suggesting actions or answering questions about the data.
        """
        formatted_prompt = prompt.format(
            analysis=analysis,
            preview=preview,
            message=message
        )
        response = rate_limited_api_call(
            client.chat.completions.create,
            model="gpt-4o",
            messages=[{"role": "user", "content": formatted_prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in chat_with_gpt: {str(e)}")
        st.error(f"Failed to process chat: {str(e)}")
        return f"Error: Failed to process chat - {str(e)}"

def suggest_workflow(df: pd.DataFrame) -> List[str]:
    if isinstance(df, bytes):
        df = decrypt_dataframe(df)

    username = st.session_state.get('username', 'anonymous')
    log_action(username, "Suggested workflow")

    if not client:
        return ["Configure a valid OpenAI API key for workflow suggestions."]

    try:
        analysis = analyze_dataset(df)
        suggestions = get_cleaning_suggestions(df)
        workflow = []
        
        for suggestion, explanation in suggestions:
            workflow.append(f"Step: {suggestion} - Reason: {explanation}")
        
        if analysis["cat_cols"]:
            workflow.append("Step: Encode categorical columns - Reason: Prepares data for ML.")
        if len(analysis["numeric_cols"]) >= 2:
            workflow.append("Step: Generate polynomial features - Reason: Enhances model performance.")
        
        if analysis["numeric_cols"]:
            workflow.append("Step: Train a predictive model - Reason: Enables forecasting.")
        
        if len(analysis["numeric_cols"]) >= 2:
            workflow.append("Step: Perform clustering - Reason: Identifies groupings.")
        
        viz_type, viz_reason = suggest_visualization(df)
        workflow.append(f"Step: Create a {viz_type} chart - Reason: {viz_reason}")
        
        return workflow
    except Exception as e:
        logger.error(f"Error in suggest_workflow: {str(e)}")
        st.error(f"Failed to suggest workflow: {str(e)}")
        return [f"Error: Failed to suggest workflow - {str(e)}"]

def apply_cleaning_operations(
    df: pd.DataFrame,
    selected_suggestions: List[Tuple[str, str]],
    columns_to_drop: List[str],
    options: Dict[str, str],
    replace_value: str,
    replace_with: str,
    replace_scope: str,
    encode_cols: List[str],
    encode_method: str,
    auto_clean: bool = False,
    enrich_col: Optional[str] = None,
    enrich_api_key: Optional[str] = None,
    train_ml: bool = False,
    target_col: Optional[str] = None,
    feature_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    if isinstance(df, bytes):
        df = decrypt_dataframe(df)

    username = st.session_state.get('username', 'anonymous')
    log_action(username, "Applied cleaning operations")

    cleaned_df = df.copy()
    logs = []

    try:
        if columns_to_drop:
            cleaned_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
            logs.append(f"Dropped columns: {columns_to_drop}")

        if replace_value and replace_with is not None:
            if not replace_value.strip():
                logs.append("No value provided for replacement.")
            else:
                target_cols = (
                    cleaned_df.columns if replace_scope == "All columns" else
                    cleaned_df.select_dtypes(include=['int64', 'float64']).columns if replace_scope == "Numeric columns" else
                    cleaned_df.select_dtypes(include=['object', 'category']).columns
                )
                replace_count = 0
                for col in target_cols:
                    try:
                        col_values = cleaned_df[col].astype(str).str.lower()
                        replace_value_lower = str(replace_value).lower()
                        matches = col_values == replace_value_lower
                        replace_count += matches.sum()
                        if replace_with.lower() == "nan":
                            cleaned_df.loc[matches, col] = np.nan
                        else:
                            cleaned_df.loc[matches, col] = replace_with
                        logger.info(f"Column {col}: Replaced {matches.sum()} instances of '{replace_value}' with '{replace_with}'")
                    except Exception as e:
                        logger.error(f"Error replacing value in column {col}: {str(e)}")
                        logs.append(f"Failed to replace '{replace_value}' in column {col}: {str(e)}")
                logs.append(f"Replaced '{replace_value}' with '{replace_with}' in {replace_scope} ({replace_count} instances)" if replace_count > 0 else
                            f"No instances of '{replace_value}' found in {replace_scope}")

        if encode_cols:
            le = LabelEncoder()
            for col in encode_cols:
                if col in cleaned_df.columns and cleaned_df[col].dtype in ['object', 'category']:
                    if encode_method == "Label Encoding":
                        cleaned_df[col] = le.fit_transform(cleaned_df[col].astype(str))
                        logs.append(f"Converted {col} to numerical using Label Encoding")
                    elif encode_method == "One-Hot Encoding":
                        cleaned_df = pd.get_dummies(cleaned_df, columns=[col], drop_first=True)
                        logs.append(f"Converted {col} to numerical using One-Hot Encoding")
                else:
                    logs.append(f"Column {col} not found or not categorical for encoding")

        if enrich_col:
            cleaned_df, enrich_log = enrich_with_geolocation(cleaned_df, enrich_col, enrich_api_key)
            logs.append(enrich_log)

        # Fix 25: Respect selected_suggestions unless auto_clean is True
        suggestions_to_apply = get_cleaning_suggestions(df) if auto_clean else selected_suggestions
        for suggestion, explanation in suggestions_to_apply:
            if "Replace '?' with NaN" in suggestion:
                if '?' in cleaned_df.values:
                    cleaned_df.replace('?', np.nan, inplace=True)
                    logs.append(f"Replaced all '?' with NaN - {explanation}")
                else:
                    logs.append(f"No '?' found to replace - {explanation}")
            
            elif "Handle special characters" in suggestion:
                special_cols = [col for col in cleaned_df.columns if any(c in col for c in "#@$%^&* ()")]
                if special_cols:
                    choice = options.get("special_chars", "Drop them")
                    if choice == "Drop them":
                        cleaned_df.drop(columns=special_cols, inplace=True)
                        logs.append(f"Dropped columns with special characters: {special_cols} - {explanation}")
                    else:
                        cleaned_df.columns = [re.sub(r'[#@$%^&* ()]', '_', col) for col in cleaned_df.columns]
                        logs.append(f"Replaced special characters with underscores in column names - {explanation}")
                else:
                    logs.append(f"No special character columns found - {explanation}")
            
            elif "Remove fully empty rows" in suggestion:
                empty_rows = cleaned_df.isna().all(axis=1)
                if empty_rows.any():
                    cleaned_df = cleaned_df[~empty_rows]
                    logs.append(f"Dropped {empty_rows.sum()} empty rows - {explanation}")
                else:
                    logs.append(f"No fully empty rows found - {explanation}")
            
            elif "Fill missing values" in suggestion:
                col = extract_column(suggestion)
                if col and col in cleaned_df.columns and cleaned_df[col].isna().any():
                    method = options.get(f"fill_{col}", "mode")
                    if cleaned_df[col].dtype in ['int64', 'float64']:
                        if method == "mean":
                            cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
                            logs.append(f"Filled missing values in {col} with mean - {explanation}")
                        elif method == "median":
                            cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
                            logs.append(f"Filled missing values in {col} with median - {explanation}")
                        else:
                            cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
                            logs.append(f"Filled missing values in {col} with mode - {explanation}")
                    else:
                        cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
                        logs.append(f"Filled missing values in {col} with mode - {explanation}")
                else:
                    logs.append(f"No missing values to fill in {col or 'specified column'} - {explanation}")
            
            elif "Encode categorical column" in suggestion:
                col = extract_column(suggestion)
                if col and col in cleaned_df.columns and cleaned_df[col].dtype in ['object', 'category']:
                    cleaned_df = pd.get_dummies(cleaned_df, columns=[col], drop_first=True)
                    logs.append(f"Encoded categorical column: {col} - {explanation}")
                else:
                    logs.append(f"No categorical column {col or 'specified'} to encode - {explanation}")
            
            elif "Remove duplicate rows" in suggestion:
                initial_rows = len(cleaned_df)
                cleaned_df.drop_duplicates(inplace=True)
                rows_dropped = initial_rows - len(cleaned_df)
                if rows_dropped > 0:
                    logs.append(f"Removed {rows_dropped} duplicate rows - {explanation}")
                else:
                    logs.append(f"No duplicate rows found - {explanation}")
            
            elif "Handle outliers" in suggestion:
                col = extract_column(suggestion)
                if col and col in cleaned_df.columns and cleaned_df[col].dtype in ['int64', 'float64']:
                    has_outliers, lower, upper = detect_outliers(cleaned_df, col)
                    if has_outliers:
                        action = options.get(f"outlier_{col}", "Remove")
                        if action == "Remove":
                            cleaned_df = cleaned_df[(cleaned_df[col] >= lower) & (cleaned_df[col] <= upper)]
                            logs.append(f"Removed outliers in {col} - {explanation}")
                        else:
                            cleaned_df[col] = cleaned_df[col].clip(lower, upper)
                            logs.append(f"Capped outliers in {col} - {explanation}")
                    else:
                        logs.append(f"No outliers in {col} - {explanation}")
                else:
                    logs.append(f"No numeric column {col or 'specified'} for outlier handling - {explanation}")
            
            elif "Interpolate time series" in suggestion:
                col = extract_column(suggestion)
                if col and col in cleaned_df.columns and pd.api.types.is_datetime64_any_dtype(cleaned_df.index):
                    cleaned_df = interpolate_time_series(cleaned_df, col)
                    logs.append(f"Interpolated time series in {col} - {explanation}")
            
            elif "Convert column" in suggestion:
                col = extract_column(suggestion)
                if col and col in cleaned_df.columns:
                    suggested_type = re.search(r"to\s+(\w+)", suggestion).group(1).lower()
                    try:
                        if suggested_type == "numeric":
                            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                            logs.append(f"Converted column {col} to numeric - {explanation}")
                        elif suggested_type == "string":
                            cleaned_df[col] = cleaned_df[col].astype(str)
                            logs.append(f"Converted column {col} to string - {explanation}")
                    except Exception as e:
                        logs.append(f"Failed to convert column {col} to {suggested_type}: {str(e)} - {explanation}")

        if train_ml and target_col and feature_cols:
            cleaned_df = auto_feature_engineering(cleaned_df, feature_cols)
            feature_cols = [col for col in cleaned_df.columns if col != target_col]
            model, score, explainer, shap_values, X_test = train_ml_model(cleaned_df, target_col, feature_cols, task_type="classification")
            if score is not None:
                app_path = generate_ml_app(cleaned_df, target_col, feature_cols)
                logs.append(f"Trained ML model with accuracy {score:.2f}. Generated app at {app_path}")
            else:
                logs.append("ML model training failed.")

        return cleaned_df, logs
    except Exception as e:
        logger.error(f"Error in apply_cleaning_operations: {str(e)}")
        st.error(f"Cleaning operations failed: {str(e)}")
        return df, [f"Error: Cleaning operations failed - {str(e)}"]