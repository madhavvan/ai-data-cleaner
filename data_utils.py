import pandas as pd
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv
import re
from sklearn.preprocessing import LabelEncoder
import requests
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

def detect_outliers(df, col):
    """Detect outliers in a numeric column using IQR method."""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    return len(outliers) > 0, lower_bound, upper_bound

def analyze_dataset(df):
    """Analyze dataset properties for GPT suggestions and health score."""
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

def calculate_health_score(df):
    """Calculate a dataset health score (0-100) based on quality metrics."""
    analysis = analyze_dataset(df)
    score = 100
    if analysis["has_question_marks"]: score -= 10
    if analysis["special_char_cols"]: score -= 5 * len(analysis["special_char_cols"])
    if analysis["empty_rows"]: score -= min(20, analysis["empty_rows"] * 2)
    if analysis["missing_cols"]: score -= min(30, len(analysis["missing_cols"]) * 5)
    if analysis["duplicates"]: score -= min(20, analysis["duplicates"] * 2)
    return max(0, score)

def get_cleaning_suggestions(df):
    """Generate AI-driven cleaning suggestions with explanations using GPT-4o."""
    if not client:
        return [("Error: OpenAI API key not configured", "API key missing")]
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

def extract_column(suggestion):
    """Extract column name from a suggestion string."""
    match = re.search(r"in\s+['\"]?(.*?)['\"]?\s*(?:with|$)", suggestion)
    return match.group(1) if match else None

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
        return df, f"Geolocation enrichment failed: {str(e)}"

def interpolate_time_series(df, col):
    """Interpolate missing values in a time series column."""
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].interpolate(method='linear')
    return df

def forecast_time_series(df, col, periods=5):
    """Forecast future values for a time series column."""
    model = ARIMA(df[col].dropna(), order=(1, 1, 1))
    fitted = model.fit()
    forecast = fitted.forecast(steps=periods)
    forecast_df = pd.DataFrame({col: forecast}, index=pd.date_range(start=df[col].index[-1], periods=periods+1, freq='D')[1:])
    return forecast_df

def train_ml_model(df, target_col, feature_cols):
    """Train a simple ML model and save it."""
    X = df[feature_cols].fillna(0)
    y = df[target_col].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    joblib.dump(model, "model.pkl")
    return score

def generate_ml_app(df, target_col, feature_cols):
    """Generate a Streamlit app script for the trained model."""
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

def chat_with_gpt(df, message):
    """Chat with GPT about the dataset."""
    if not client:
        return "Error: OpenAI API key not configured"
    analysis = analyze_dataset(df)
    prompt = f"""
    You are an AI data assistant. Respond to this user message based on the dataset analysis:
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

def apply_cleaning_operations(df, selected_suggestions, columns_to_drop, options, replace_value, replace_with, replace_scope, encode_cols, encode_method, auto_clean=False, enrich_col=None, enrich_api_key=None, train_ml=False, target_col=None, feature_cols=None):
    """Apply all selected cleaning operations to the dataset."""
    cleaned_df = df.copy()
    logs = []
    
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
        score = train_ml_model(cleaned_df, target_col, feature_cols)
        app_path = generate_ml_app(cleaned_df, target_col, feature_cols)
        logs.append(f"Trained ML model with accuracy {score:.2f}. Generated app at {app_path}")
    
    return cleaned_df, logs