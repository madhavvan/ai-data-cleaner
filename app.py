import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
import io
import base64
from datetime import datetime
import os
from dotenv import load_dotenv
import re
from sklearn.preprocessing import LabelEncoder

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found. Please set it in the `.env` file as `OPENAI_API_KEY`.")
    st.stop()
client = OpenAI(api_key=api_key)

# --- Data Processing Functions ---

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
    """Analyze dataset properties for GPT suggestions."""
    return {
        "has_question_marks": '?' in df.values,
        "special_char_cols": [col for col in df.columns if any(c in col for c in "#@$%^&* ()")],
        "empty_rows": df.isna().all(axis=1).sum(),
        "missing_cols": df.columns[df.isna().any()].tolist(),
        "numeric_cols": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        "cat_cols": df.select_dtypes(include=['object', 'category']).columns.tolist(),
        "duplicates": df.duplicated().sum()
    }

@st.cache_data
def get_cleaning_suggestions(df):
    """Generate AI-driven cleaning suggestions using GPT-4o."""
    analysis = analyze_dataset(df)
    prompt = f"""
    You are an expert data analyst. Based on this dataset analysis, provide specific, actionable cleaning suggestions:
    - Dataset preview (first 10 rows): {df.head(10).to_string()}
    - Analysis:
      - '?' present: {analysis['has_question_marks']}
      - Columns with special characters: {analysis['special_char_cols']}
      - Fully empty rows: {analysis['empty_rows']}
      - Columns with missing values: {analysis['missing_cols']}
      - Numeric columns: {analysis['numeric_cols']}
      - Categorical columns: {analysis['cat_cols']}
      - Duplicate rows: {analysis['duplicates']}
    
    Suggest only applicable operations with specific wording:
    1. "Replace '?' with NaN" if '?' exists.
    2. "Handle special characters in columns: [list]" if special chars exist.
    3. "Remove fully empty rows" if empty rows exist.
    4. "Fill missing values in [col] with [mean/median/mode]" for each column with missing values.
    5. "Encode categorical column: [col]" for each categorical column.
    6. "Remove duplicate rows" if duplicates exist.
    7. "Handle outliers in [col]" for each numeric column with outliers.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    suggestions = response.choices[0].message.content.strip().split("\n")
    return [s.strip("1234567890. ") for s in suggestions if s.strip()]

def extract_column(suggestion):
    """Extract column name from a suggestion string."""
    match = re.search(r"in\s+['\"]?(.*?)['\"]?\s*(?:with|$)", suggestion)
    return match.group(1) if match else None

def apply_cleaning_operations(df, selected_suggestions, columns_to_drop, options, replace_value, replace_with, replace_scope, encode_cols, encode_method):
    """Apply all selected cleaning operations to the dataset."""
    cleaned_df = df.copy()
    logs = []
    
    # Manual column dropping
    if columns_to_drop:
        cleaned_df.drop(columns=columns_to_drop, inplace=True)
        logs.append(f"Dropped columns: {columns_to_drop}")
    
    # Custom value replacement
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
    
    # Categorical to numerical conversion
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
    
    # AI-suggested cleaning
    for suggestion in selected_suggestions:
        if "Replace '?' with NaN" in suggestion:
            if '?' in cleaned_df.values:
                cleaned_df.replace('?', np.nan, inplace=True)
                logs.append("Replaced all '?' with NaN")
            else:
                logs.append("No '?' found to replace")
        
        elif "Handle special characters" in suggestion:
            special_cols = [col for col in cleaned_df.columns if any(c in col for c in "#@$%^&* ()")]
            if special_cols:
                choice = options.get("special_chars", "Drop them")
                if choice == "Drop them":
                    cleaned_df.drop(columns=special_cols, inplace=True)
                    logs.append(f"Dropped columns with special characters: {special_cols}")
                else:
                    cleaned_df.columns = [''.join('_' if c in "#@$%^&* ()" else c for c in col) 
                                        for col in cleaned_df.columns]
                    logs.append("Replaced special characters with underscores")
            else:
                logs.append("No special character columns found")
        
        elif "Remove fully empty rows" in suggestion:
            empty_rows = cleaned_df.isna().all(axis=1)
            if empty_rows.any():
                cleaned_df = cleaned_df[~empty_rows]
                logs.append(f"Dropped {empty_rows.sum()} empty rows")
            else:
                logs.append("No fully empty rows found")
        
        elif "Fill missing values" in suggestion:
            col = extract_column(suggestion)
            if col and col in cleaned_df.columns and cleaned_df[col].isna().any():
                method = options.get(f"fill_{col}", "mode")
                if cleaned_df[col].dtype in ['int64', 'float64']:
                    if method == "mean":
                        cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
                        logs.append(f"Filled missing values in {col} with mean")
                    elif method == "median":
                        cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
                        logs.append(f"Filled missing values in {col} with median")
                    else:
                        cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
                        logs.append(f"Filled missing values in {col} with mode")
                else:
                    cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
                    logs.append(f"Filled missing values in {col} with mode")
            else:
                logs.append(f"No missing values to fill in {col or 'specified column'}")
        
        elif "Encode categorical column" in suggestion:
            col = extract_column(suggestion)
            if col and col in cleaned_df.columns and cleaned_df[col].dtype in ['object', 'category']:
                cleaned_df = pd.get_dummies(cleaned_df, columns=[col], drop_first=True)
                logs.append(f"Encoded categorical column: {col}")
            else:
                logs.append(f"No categorical column {col or 'specified'} to encode")
        
        elif "Remove duplicate rows" in suggestion:
            initial_rows = len(cleaned_df)
            cleaned_df.drop_duplicates(inplace=True)
            rows_dropped = initial_rows - len(cleaned_df)
            if rows_dropped > 0:
                logs.append(f"Removed {rows_dropped} duplicate rows")
            else:
                logs.append("No duplicate rows found")
        
        elif "Handle outliers" in suggestion:
            col = extract_column(suggestion)
            if col and col in cleaned_df.columns and cleaned_df[col].dtype in ['int64', 'float64']:
                has_outliers, lower, upper = detect_outliers(cleaned_df, col)
                if has_outliers:
                    action = options.get(f"outlier_{col}", "Remove")
                    if action == "Remove":
                        cleaned_df = cleaned_df[(cleaned_df[col] >= lower) & (cleaned_df[col] <= upper)]
                        logs.append(f"Removed outliers in {col}")
                    else:
                        cleaned_df[col] = cleaned_df[col].clip(lower, upper)
                        logs.append(f"Capped outliers in {col}")
                else:
                    logs.append(f"No outliers in {col}")
            else:
                logs.append(f"No numeric column {col or 'specified'} for outlier handling")
    
    return cleaned_df, logs

def get_download_link(df, filename):
    """Generate a download link for the cleaned dataset."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download cleaned dataset</a>'

# --- UI Rendering ---

def render_upload_page():
    """Render the upload page UI."""
    st.title("📤 Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"], help="Upload a CSV or Excel file to begin.")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.session_state.cleaned_df = None
            st.session_state.logs = []
            st.session_state.suggestions = []
            st.session_state.previous_states = []
            
            st.subheader("Dataset Preview (First 10 Rows)")
            st.dataframe(df.head(10))
            
            st.subheader("Basic Metadata")
            st.write(f"Rows: {df.shape[0]}")
            st.write(f"Columns: {df.shape[1]}")
            st.write(f"Missing Values: {df.isna().sum().sum()}")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

def render_clean_page():
    """Render the clean page UI."""
    st.title("🧹 Clean Your Dataset")
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("Please upload a dataset first on the Upload page.")
        return
    
    df = st.session_state.df
    
    # Generate suggestions once
    if not st.session_state.suggestions:
        with st.spinner("Analyzing dataset with GPT-4o..."):
            st.session_state.suggestions = get_cleaning_suggestions(df)
    
    with st.form(key="cleaning_form"):
        # Use containers for better organization
        manual_container = st.container()
        replace_container = st.container()
        encode_container = st.container()
        ai_container = st.container()
        
        # Manual Column Dropping
        with manual_container:
            st.subheader("Manual Column Dropping")
            columns_to_drop = st.multiselect("Select columns to drop", df.columns.tolist(), 
                                           help="Choose columns to remove from the dataset.")
        
        # Custom Value Replacement
        with replace_container:
            with st.expander("Custom Value Replacement", expanded=False):
                replace_value = st.text_input("Value to replace (e.g., 999, Unknown)", "", 
                                            help="Enter the value you want to replace.")
                replace_with = st.radio("Replace with", ["NaN", "?", "0", "Custom"], 
                                      help="Select what to replace the value with.")
                if replace_with == "Custom":
                    replace_with = st.text_input("Custom replacement value", "", 
                                               help="Enter a custom replacement value.")
                replace_scope = st.radio("Apply to", ["All columns", "Numeric columns", "Categorical columns"], 
                                       help="Choose which columns to apply the replacement to.")
        
        # Categorical to Numerical Conversion
        with encode_container:
            with st.expander("Convert Categorical to Numerical", expanded=False):
                cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                encode_cols = st.multiselect("Select categorical columns to convert", cat_cols, 
                                           help="Choose categorical columns to convert to numerical.")
                encode_method = st.radio("Conversion method", ["Label Encoding", "One-Hot Encoding"], 
                                       help="Label Encoding assigns integers; One-Hot creates dummy columns.")
        
        # AI Cleaning Suggestions
        with ai_container:
            with st.expander("AI Cleaning Suggestions", expanded=True):
                selected_suggestions = []
                options = {}
                for suggestion in st.session_state.suggestions:
                    if st.checkbox(suggestion, key=suggestion, help=f"This will {suggestion.lower()}"):
                        selected_suggestions.append(suggestion)
                        if "Handle special characters" in suggestion:
                            options["special_chars"] = st.radio("Action for special characters", 
                                                              ("Drop them", "Replace with underscores"), 
                                                              key="special_chars_opt")
                        elif "Fill missing values" in suggestion:
                            col = extract_column(suggestion)
                            if col and col in df.columns and df[col].dtype in ['int64', 'float64']:
                                options[f"fill_{col}"] = st.radio(f"Fill method for {col}", 
                                                                ["mean", "median", "mode"], 
                                                                key=f"fill_opt_{col}")
                        elif "Handle outliers" in suggestion:
                            col = extract_column(suggestion)
                            if col and col in df.columns:
                                options[f"outlier_{col}"] = st.radio(f"Action for outliers in {col}", 
                                                                   ("Remove", "Cap at bounds"), 
                                                                   key=f"outlier_opt_{col}")
        
        # Form Buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            preview_button = st.form_submit_button(label="Preview Changes")
        with col2:
            apply_button = st.form_submit_button(label="Apply Changes")
        with col3:
            reset_button = st.form_submit_button(label="Reset Form")
    
    # Handle Reset
    if reset_button:
        st.session_state.pop('cleaning_form', None)  # Clear form state
        st.rerun()
    
    # Process Preview or Apply
    if (preview_button or apply_button) and (selected_suggestions or columns_to_drop or replace_value or encode_cols):
        with st.spinner("Processing..."):
            cleaned_df, logs = apply_cleaning_operations(df, selected_suggestions, columns_to_drop, options, 
                                                       replace_value, replace_with if replace_with != "NaN" else "NaN", 
                                                       replace_scope, encode_cols, encode_method)
            
            if preview_button:
                st.subheader("Preview of Changes")
                st.write("Before:")
                st.dataframe(df.head(10))
                st.write("After:")
                st.dataframe(cleaned_df.head(10))
                st.write("Preview Logs:")
                for log in logs:
                    st.write(f"- {log}")
            
            if apply_button:
                if st.session_state.cleaned_df is not None:
                    st.session_state.previous_states.append((st.session_state.cleaned_df.copy(), st.session_state.logs.copy()))
                else:
                    st.session_state.previous_states.append((df.copy(), []))
                if len(st.session_state.previous_states) > 5:
                    st.session_state.previous_states.pop(0)
                
                st.session_state.cleaned_df = cleaned_df
                st.session_state.logs = logs
    
    # Undo Button
    if st.session_state.get('previous_states') and st.button("Undo Last Cleaning", help="Revert to the previous state"):
        previous_df, previous_logs = st.session_state.previous_states.pop()
        st.session_state.cleaned_df = previous_df
        st.session_state.logs = previous_logs
    
    # Display Results
    if st.session_state.get('cleaned_df') is not None:
        st.subheader("Cleaned Dataset Preview")
        st.dataframe(st.session_state.cleaned_df.head(10))
        
        st.subheader("Cleaning Summary")
        st.write(f"Original Shape: {df.shape}")
        st.write(f"New Shape: {st.session_state.cleaned_df.shape}")
        for log in st.session_state.logs:
            st.write(f"- {log}")
        
        st.markdown(get_download_link(st.session_state.cleaned_df, 
                                    f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), 
                   unsafe_allow_html=True)

# --- Main App ---

st.set_page_config(page_title="AI Data Cleaner", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload", "Clean"])

if page == "Upload":
    render_upload_page()
elif page == "Clean":
    render_clean_page()

# Add some styling
st.markdown("""
<style>
    .stCheckbox { margin-bottom: 10px; }
    .stRadio { margin-bottom: 10px; }
    .stSelectbox { margin-bottom: 10px; }
    .stTextInput { margin-bottom: 10px; }
    .stButton { margin-top: 10px; }
</style>
""", unsafe_allow_html=True)