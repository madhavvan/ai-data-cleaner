import pandas as pd
import numpy as np
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
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable proxies at the environment level as a precaution
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "*"

# Load OpenAI API key from Streamlit secrets with fallback to environment variable
try:
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    logger.info("Successfully loaded OPENAI_API_KEY from secrets")
except FileNotFoundError:
    logger.warning("No secrets.toml file found. Checking environment variable OPENAI_API_KEY.")
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        logger.info("Successfully loaded OPENAI_API_KEY from environment variable")
    else:
        logger.warning("OpenAI API key not found in environment variable either.")
except Exception as e:
    logger.error(f"Error loading secrets: {str(e)}")
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        logger.info("Successfully loaded OPENAI_API_KEY from environment variable")
    else:
        logger.warning("OpenAI API key not found in environment variable either.")

# Initialize OpenAI client with minimal configuration
client = None
if api_key:
    try:
        client = OpenAI(api_key=api_key)  # Simplified initialization
        logger.info("OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {str(e)}")
        client = None
else:
    logger.warning("OpenAI API key not found. AI-driven features will be disabled.")

def detect_outliers(df, col):
    """Detect outliers in a numeric column using IQR method."""
    try:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        return len(outliers) > 0, lower_bound, upper_bound
    except Exception as e:
        logger.error(f"Error in detect_outliers for column {col}: {str(e)}")
        return False, 0, 0

def detect_anomalies(df, cols):
    """Detect anomalies in numerical columns using Isolation Forest."""
    anomalies = {}
    try:
        for col in cols:
            data = df[[col]].dropna()
            if not data.empty:
                model = IsolationForest(contamination=0.1, random_state=42)
                predictions = model.fit_predict(data)
                anomaly_indices = data[predictions == -1].index
                anomalies[col] = df.loc[anomaly_indices, col].to_dict()
            else:
                logger.warning(f"No data available for anomaly detection in column {col}")
    except Exception as e:
        logger.error(f"Error in detect_anomalies: {str(e)}")
    return anomalies

def analyze_dataset(df):
    """Analyze dataset properties for GPT suggestions and health score."""
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
        return analysis
    except Exception as e:
        logger.error(f"Error in analyze_dataset: {str(e)}")
        return {}

def calculate_health_score(df):
    """Calculate a dataset health score (0-100) based on quality metrics."""
    try:
        analysis = analyze_dataset(df)
        score = 100
        if analysis.get("has_question_marks"): score -= 10
        if analysis.get("special_char_cols"): score -= 5 * len(analysis["special_char_cols"])
        if analysis.get("empty_rows"): score -= min(20, analysis["empty_rows"] * 2)
        if analysis.get("missing_cols"): score -= min(30, len(analysis["missing_cols"]) * 5)
        if analysis.get("duplicates"): score -= min(20, analysis["duplicates"] * 2)
        return max(0, score)
    except Exception as e:
        logger.error(f"Error in calculate_health_score: {str(e)}")
        return 0

def get_cleaning_suggestions(df):
    """Generate AI-driven cleaning suggestions with explanations using GPT-4o."""
    if not client:
        return [("Error: OpenAI API key not configured", "API key missing or client initialization failed")]
    try:
        analysis = analyze_dataset(df)
        prompt = f"""
        You are an expert data analyst. Based on this dataset analysis, provide specific, actionable cleaning suggestions with brief explanations:
        - Dataset preview (first 10 rows): {df.head(10).to_string()}
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
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700
        )
        suggestions = response.choices[0].message.content.strip().split("\n")
        return [(s.split(" - ")[0].strip("1234567890. "), s.split(" - ")[1] if " - " in s else "No explanation provided") 
                for s in suggestions if s.strip()]
    except Exception as e:
        logger.error(f"Error in get_cleaning_suggestions: {str(e)}")
        return [("Error: Failed to generate suggestions", str(e))]

def get_insights(df):
    """Generate natural language insights about the dataset."""
    if not client:
        return "Error: OpenAI API key not configured or client initialization failed"
    try:
        prompt = f"""
        You are an AI data analyst. Analyze this dataset and provide 3-5 human-readable insights in plain English:
        - Dataset preview (first 10 rows): {df.head(10).to_string()}
        - Analysis: {analyze_dataset(df)}
        Examples of insights:
        - "Column X has a strong correlation with Column Y, suggesting a potential relationship."
        - "Sales increased by 20% in Q3, driven by Region A."
        - "30% of the data in Column Z is missing, which may impact analysis."
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        return response.choices[0].message.content.strip().split("\n")
    except Exception as e:
        logger.error(f"Error in get_insights: {str(e)}")
        return [f"Error: Failed to generate insights - {str(e)}"]

def suggest_visualization(df):
    """Suggest the best visualization type based on data characteristics."""
    try:
        analysis = analyze_dataset(df)
        if analysis["time_cols"]:
            return "Line", "Visualize trends over time with a line chart."
        elif len(analysis["numeric_cols"]) >= 2:
            return "Scatter", "Explore relationships between numerical variables with a scatter plot."
        elif len(analysis["cat_cols"]) > 0 and len(analysis["numeric_cols"]) > 0:
            return "Bar", "Compare categories with a bar chart."
        else:
            return "Histogram", "Understand the distribution of a numerical column with a histogram."
    except Exception as e:
        logger.error(f"Error in suggest_visualization: {str(e)}")
        return "Bar", "Default suggestion due to error."

def extract_column(suggestion):
    """Extract column name from a suggestion string."""
    try:
        match = re.search(r"in\s+['\"]?(.*?)['\"]?\s*(?:with|$)", suggestion)
        return match.group(1) if match else None
    except Exception as e:
        logger.error(f"Error in extract_column: {str(e)}")
        return None

def enrich_with_geolocation(df, address_col, api_key=None):
    """Enrich dataset with geolocation data."""
    if not api_key:
        return df, "No Google API key provided"
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
        return df, f"Enriched {address_col} with lat/lon"
    except Exception as e:
        logger.error(f"Error in enrich_with_geolocation: {str(e)}")
        return df, f"Geolocation enrichment failed: {str(e)}"

def interpolate_time_series(df, col):
    """Interpolate missing values in a time series column."""
    try:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].interpolate(method='linear')
        return df
    except Exception as e:
        logger.error(f"Error in interpolate_time_series: {str(e)}")
        return df

def analyze_time_series(df, col, period=12):
    """Analyze time series for trends, seasonality, and residuals."""
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
        return {}

def forecast_time_series(df, col, periods=5):
    """Forecast future values for a time series column."""
    try:
        model = ARIMA(df[col].dropna(), order=(1, 1, 1))
        fitted = model.fit()
        forecast = fitted.forecast(steps=periods)
        forecast_df = pd.DataFrame({col: forecast}, index=pd.date_range(start=df.index[-1], periods=periods+1, freq='D')[1:])
        return forecast_df
    except Exception as e:
        logger.error(f"Error in forecast_time_series: {str(e)}")
        return pd.DataFrame()

def generate_synthetic_data(df, task_type="classification"):
    """Generate synthetic data based on the dataset's structure."""
    try:
        n_samples = len(df)
        n_features = len(df.columns) - 1  # Assume last column is target
        if task_type == "classification":
            X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_features-2, random_state=42)
        else:
            X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=0.1, random_state=42)
        
        synthetic_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
        synthetic_df["target"] = y
        return synthetic_df
    except Exception as e:
        logger.error(f"Error in generate_synthetic_data: {str(e)}")
        return pd.DataFrame()

def auto_feature_engineering(df, feature_cols, degree=2):
    """Automatically generate new features (e.g., polynomial features)."""
    try:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X = df[feature_cols].fillna(0)
        poly_features = poly.fit_transform(X)
        poly_feature_names = poly.get_feature_names_out(feature_cols)
        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
        return pd.concat([df.drop(columns=feature_cols), poly_df], axis=1)
    except Exception as e:
        logger.error(f"Error in auto_feature_engineering: {str(e)}")
        return df

def train_ml_model(df, target_col, feature_cols, task_type="classification"):
    """Train an ML model with hyperparameter tuning and return model, score, and SHAP explainer."""
    try:
        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if task_type == "classification":
            model = RandomForestClassifier(random_state=42)
            param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20, None]}
        else:
            model = RandomForestRegressor(random_state=42)
            param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20, None]}
        
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy' if task_type == "classification" else 'r2')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        score = best_model.score(X_test, y_test)
        
        # Attempt to import and use SHAP, with fallback if not available
        try:
            import shap
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test)
        except ImportError:
            logger.warning("SHAP library not installed. Feature importance plots will not be available.")
            explainer = None
            shap_values = None
        
        joblib.dump(best_model, "model.pkl")
        return best_model, score, explainer, shap_values, X_test
    except Exception as e:
        logger.error(f"Error in train_ml_model: {str(e)}")
        return None, 0, None, None, None

def perform_clustering(df, feature_cols, n_clusters=3):
    """Perform clustering on the dataset."""
    try:
        X = df[feature_cols].fillna(0)
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X)
        return labels
    except Exception as e:
        logger.error(f"Error in perform_clustering: {str(e)}")
        return np.zeros(len(df))

def generate_ml_app(df, target_col, feature_cols):
    """Generate a Streamlit app script for the trained model."""
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
        return f"Error: Failed to generate ML app - {str(e)}"

def chat_with_gpt(df, message):
    """Chat with GPT about the dataset, with identity response for relevant questions."""
    if not client:
        return "Error: OpenAI API key not configured or client initialization failed"
    
    # Check for identity-related questions
    identity_keywords = ["who are you", "what are you", "who created you", "what's your name"]
    if any(keyword in message.lower() for keyword in identity_keywords):
        return "I’m Madhavvan’s personal training assistant, built for data analysis. How can I assist you today?"
    
    try:
        analysis = analyze_dataset(df)
        prompt = f"""
        You are Madhavvan's personal training assistant, an AI built for data analysis. Respond to this user message based on the dataset analysis:
        - Analysis: {analysis}
        - Dataset preview (first 10 rows): {df.head(10).to_string()}
        User message: "{message}"
        Provide a helpful response, suggesting actions or answering questions about the data.
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in chat_with_gpt: {str(e)}")
        return f"Error: Failed to process chat - {str(e)}"

def suggest_workflow(df):
    """Suggest an automated workflow for the dataset."""
    try:
        analysis = analyze_dataset(df)
        suggestions = get_cleaning_suggestions(df)
        workflow = []
        
        # Add cleaning steps
        for suggestion, explanation in suggestions:
            workflow.append(f"Step: {suggestion} - Reason: {explanation}")
        
        # Add feature engineering if categorical columns exist
        if analysis["cat_cols"]:
            workflow.append("Step: Encode categorical columns - Reason: Prepares data for ML modeling.")
        if len(analysis["numeric_cols"]) >= 2:
            workflow.append("Step: Generate polynomial features - Reason: Enhances model performance.")
        
        # Add predictive modeling if numerical columns exist
        if analysis["numeric_cols"]:
            workflow.append("Step: Train a predictive model - Reason: Enables forecasting and insights.")
        
        # Add clustering if multiple features
        if len(analysis["numeric_cols"]) >= 2:
            workflow.append("Step: Perform clustering - Reason: Identifies natural groupings in the data.")
        
        # Add visualization suggestion
        viz_type, viz_reason = suggest_visualization(df)
        workflow.append(f"Step: Create a {viz_type} chart - Reason: {viz_reason}")
        
        return workflow
    except Exception as e:
        logger.error(f"Error in suggest_workflow: {str(e)}")
        return [f"Error: Failed to suggest workflow - {str(e)}"]

def apply_cleaning_operations(df, selected_suggestions, columns_to_drop, options, replace_value, replace_with, replace_scope, encode_cols, encode_method, auto_clean=False, enrich_col=None, enrich_api_key=None, train_ml=False, target_col=None, feature_cols=None):
    """Apply all selected cleaning operations to the dataset."""
    cleaned_df = df.copy()
    logs = []
    
    try:
        if columns_to_drop:
            cleaned_df.drop(columns=columns_to_drop, inplace=True)
            logs.append(f"Dropped columns: {columns_to_drop}")
        
        if replace_value and replace_with is not None:
            if not replace_value.strip():
                logs.append("No value provided for replacement")
            else:
                target_cols = (
                    cleaned_df.columns if replace_scope == "All columns" else
                    cleaned_df.select_dtypes(include=['int64', 'float64']).columns if replace_scope == "Numeric columns" else
                    cleaned_df.select_dtypes(include=['object', 'category']).columns
                )
                replace_count = 0
                for col in target_cols:
                    if replace_with == "NaN":
                        replace_count += cleaned_df[col].eq(replace_value).sum()
                        cleaned_df[col] = cleaned_df[col].replace(replace_value, np.nan)
                    else:
                        replace_count += cleaned_df[col].eq(replace_value).sum()
                        cleaned_df[col] = cleaned_df[col].replace(replace_value, replace_with)
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
        
        suggestions_to_apply = [(s, e) for s, e in get_cleaning_suggestions(df)] if auto_clean else selected_suggestions
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
                        cleaned_df.columns = [''.join('_' if c in "#@$%^&* ()" else c for c in col) 
                                            for col in cleaned_df.columns]
                        logs.append(f"Replaced special characters with underscores - {explanation}")
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
        
        if train_ml and target_col and feature_cols:
            # Apply feature engineering before training
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
        return df, [f"Error: Cleaning operations failed - {str(e)}"]