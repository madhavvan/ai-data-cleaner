import streamlit as st
import pandas as pd
from datetime import datetime
import base64
import json
import os
from typing import Optional, List, Dict, Tuple
from data_utils import (
    get_cleaning_suggestions, apply_cleaning_operations, extract_column, 
    calculate_health_score, chat_with_gpt, detect_anomalies, get_insights, 
    suggest_workflow, train_ml_model, forecast_time_series, perform_clustering, 
    generate_synthetic_data, analyze_time_series
)
from predictive import render_predictive_page as render_predictive_page_external
import pyarrow.parquet as pq  # For Parquet file support

# Cache expensive operations
@st.cache_data
def get_cached_suggestions(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Cache AI-driven cleaning suggestions to improve performance.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        List[Tuple[str, str]]: List of (suggestion, explanation) tuples.
    """
    return get_cleaning_suggestions(df)

def get_download_link(df: pd.DataFrame, filename: str) -> str:
    """
    Generate a download link for the dataset.

    Args:
        df (pd.DataFrame): DataFrame to download.
        filename (str): Name of the file to download.

    Returns:
        str: HTML download link.
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'

def profile_dataset(df: pd.DataFrame) -> Dict[str, any]:
    """
    Profile the dataset to identify data quality issues and suggest fixes.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        Dict[str, any]: Profiling results with suggestions.
    """
    profile = {}
    for col in df.columns:
        col_profile = {}
        # Check for mixed data types
        col_types = df[col].apply(type).nunique()
        col_profile['mixed_types'] = col_types > 1
        col_profile['type_suggestion'] = f"Convert {col} to {df[col].dtype.name}" if col_types > 1 else None
        
        # Check for inconsistent formats (e.g., dates in different formats)
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            formats = df[col].dropna().apply(lambda x: x.strftime('%Y-%m-%d')).nunique()
            col_profile['inconsistent_formats'] = formats > 1
            col_profile['format_suggestion'] = "Standardize date format to YYYY-MM-DD" if formats > 1 else None
        
        # Check for high missing value percentage
        missing_percentage = df[col].isna().mean() * 100
        col_profile['missing_percentage'] = missing_percentage
        col_profile['missing_suggestion'] = f"Consider filling or dropping {col} (missing {missing_percentage:.2f}%)" if missing_percentage > 10 else None
        
        profile[col] = col_profile
    return profile

def initialize_session_state() -> None:
    """
    Initialize session state variables for the application.
    """
    defaults = {
        'df': None,
        'cleaned_df': None,
        'logs': [],
        'suggestions': [],
        'previous_states': [],
        'redo_states': [],
        'chat_history': [],
        'cleaning_history': [],
        'cleaning_templates': {},
        'is_premium': False,
        'ai_suggestions_used': 0,
        'dropped_columns': [],
        'progress': {
            "Upload": "Not Started",
            "Clean": "Not Started",
            "Insights": "Not Started",
            "Visualize": "Not Started",
            "Predictive": "Not Started",
            "Share": "Not Started"
        }
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def render_upload_page() -> None:
    """
    Render the upload page UI with session persistence, extended file format support, and data profiling.
    """
    st.title("Upload Your Dataset")
    st.markdown("<p class='welcome'>Start your data journey here!</p>", unsafe_allow_html=True)

    initialize_session_state()

    # Update progress
    st.session_state.progress["Upload"] = "In Progress"


    if st.session_state.df is not None:
        st.subheader("Original Dataset Preview (First 10 Rows)")
        st.dataframe(st.session_state.df.head(10))
        st.subheader("Basic Metadata")
        score = calculate_health_score(st.session_state.df)
        st.write(f"Rows: {st.session_state.df.shape[0]}")
        st.write(f"Columns: {st.session_state.df.shape[1]}")
        st.write(f"Missing Values: {st.session_state.df.isna().sum().sum()}")
        st.progress(score / 100)
        st.write(f"Dataset Health Score: {score}/100")
        st.info("This is the original dataset. Cleaning operations are applied to a working copy.")
        st.warning("Uploading a new file will overwrite the current dataset and reset all cleaning operations. Proceed with caution!")

    uploaded_file = st.file_uploader("Choose a file (CSV, Excel, JSON, or Parquet)", type=["csv", "xlsx", "json", "parquet"], help="Upload a dataset file to begin.")
    if uploaded_file:
        try:
            with st.spinner("Loading dataset..."):
                if uploaded_file.size > 50 * 1024 * 1024:  # 50MB
                    st.warning("File size exceeds 50MB. Using chunked processing.")
                    if uploaded_file.name.endswith('.csv'):
                        chunks = pd.read_csv(uploaded_file, chunksize=10000)
                        df_list = []
                        progress_bar = st.progress(0)
                        total_chunks = uploaded_file.size // (10000 * 100) or 1
                        for i, chunk in enumerate(chunks):
                            df_list.append(chunk)
                            progress_bar.progress(min((i + 1) / total_chunks, 1.0))
                        df = pd.concat(df_list, ignore_index=True)
                    elif uploaded_file.name.endswith('.json'):
                        chunks = pd.read_json(uploaded_file, chunksize=10000)
                        df_list = []
                        progress_bar = st.progress(0)
                        total_chunks = uploaded_file.size // (10000 * 100) or 1
                        for i, chunk in enumerate(chunks):
                            df_list.append(chunk)
                            progress_bar.progress(min((i + 1) / total_chunks, 1.0))
                        df = pd.concat(df_list, ignore_index=True)
                    elif uploaded_file.name.endswith('.parquet'):
                        # Parquet files are typically smaller and more efficient, so we'll read directly
                        df = pq.read_table(uploaded_file).to_pandas()
                    else:
                        df = pd.read_excel(uploaded_file)
                else:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith('.json'):
                        df = pd.read_json(uploaded_file)
                    elif uploaded_file.name.endswith('.parquet'):
                        df = pq.read_table(uploaded_file).to_pandas()
                    else:
                        df = pd.read_excel(uploaded_file)

            # Validate dataset size and structure
            if df.shape[0] > 4000:
                st.info(f"Large dataset detected ({df.shape[0]} rows). Processing optimized for performance.")
            if df.empty:
                st.error("Uploaded dataset is empty. Please upload a valid file.")
                return

            # Data Profiling
            with st.spinner("Profiling dataset..."):
                profile = profile_dataset(df)
                st.subheader("Dataset Profile")
                for col, info in profile.items():
                    if any(info.values()):  # Only show columns with issues
                        st.write(f"**Column: {col}**")
                        if info['mixed_types']:
                            st.write(f"- Mixed Types Detected: {info['mixed_types']}")
                            st.write(f"  Suggestion: {info['type_suggestion']}")
                        if info.get('inconsistent_formats'):
                            st.write(f"- Inconsistent Formats: {info['inconsistent_formats']}")
                            st.write(f"  Suggestion: {info['format_suggestion']}")
                        if info['missing_percentage'] > 10:
                            st.write(f"- Missing Values: {info['missing_percentage']:.2f}%")
                            st.write(f"  Suggestion: {info['missing_suggestion']}")

            st.session_state.df = df
            st.session_state.cleaned_df = None
            st.session_state.logs = []
            st.session_state.suggestions = []
            st.session_state.previous_states = []
            st.session_state.redo_states = []
            st.session_state.chat_history = []
            st.session_state.cleaning_history = []
            st.session_state.cleaning_templates = {}
            st.session_state.ai_suggestions_used = 0
            st.session_state.dropped_columns = []

            st.subheader("Dataset Preview (First 10 Rows)")
            st.dataframe(df.head(10))
            st.subheader("Basic Metadata")
            score = calculate_health_score(df)
            st.write(f"Rows: {df.shape[0]}")
            st.write(f"Columns: {df.shape[1]}")
            st.write(f"Missing Values: {df.isna().sum().sum()}")
            st.progress(score / 100)
            st.write(f"Dataset Health Score: {score}/100")
            st.success("Dataset uploaded successfully!")
            st.session_state.progress["Upload"] = "Done"
        except Exception as e:
            st.error(f"Error loading file: {str(e)}. Please ensure the file is a valid CSV, Excel, JSON, or Parquet file.")
            st.session_state.progress["Upload"] = "Failed"

def render_clean_page() -> None:
    """
    Render the clean page UI with robust multi-change logic, custom rules engine, tooltips, and progress tracking.
    """
    st.title("Clean Your Dataset")
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("Please upload a dataset first on the Upload page.")
        return

    df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df
    available_columns = [col for col in df.columns if col not in st.session_state.dropped_columns]

    if not available_columns:
        st.error("No columns available for cleaning. Please upload a new dataset.")
        return

    # Update progress
    st.session_state.progress["Clean"] = "In Progress"

    # Display progress tracker
    st.subheader("Your Progress")
    progress_text = ""
    for step, status in st.session_state.progress.items():
        emoji = "✅" if status == "Done" else "🟡" if status == "In Progress" else "⬜"
        progress_text += f"{emoji} {step}: {status}\n"
    st.markdown(progress_text)

    if not st.session_state.suggestions or id(st.session_state.cleaned_df) != id(df):
        with st.spinner("Generating AI cleaning suggestions..."):
            st.session_state.suggestions = get_cached_suggestions(df[available_columns])

    st.subheader("Dataset Health")
    score = calculate_health_score(df)
    st.progress(score / 100)
    st.write(f"Current Health Score: {score}/100")

    st.subheader("Smart Workflow Automation")
    st.markdown('<span title="Run an AI-suggested cleaning workflow automatically">ℹ️</span>', unsafe_allow_html=True)
    if st.button("Run Smart Workflow"):
        with st.spinner("Generating and executing workflow..."):
            try:
                workflow = suggest_workflow(df[available_columns])
                st.write("### Suggested Workflow:")
                for step in workflow:
                    st.write(f"- {step}")
                cleaned_df, logs = apply_cleaning_operations(
                    df, st.session_state.suggestions, [], {}, "", "", "", [], "", auto_clean=True
                )
                st.session_state.previous_states.append((df.copy(), st.session_state.logs.copy()))
                st.session_state.cleaned_df = cleaned_df
                st.session_state.logs = logs
                st.session_state.redo_states = []
                st.session_state.cleaning_history.append({
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "logs": logs + ["Executed Smart Workflow"]
                })
                st.session_state.suggestions = get_cached_suggestions(cleaned_df[[col for col in cleaned_df.columns if col not in st.session_state.dropped_columns]])
                st.success("Smart Workflow executed successfully!")
                st.session_state.progress["Clean"] = "Done"
                st.rerun()
            except Exception as e:
                st.error(f"Error executing smart workflow: {str(e)}. Please check the dataset and try again.")
                st.session_state.progress["Clean"] = "Failed"

    with st.form(key="cleaning_form", clear_on_submit=False):
        manual_container = st.container()
        custom_rules_container = st.container()
        replace_container = st.container()
        encode_container = st.container()
        enrich_container = st.container()
        ai_container = st.container()
        anomaly_container = st.container()
        ml_container = st.container()

        selected_suggestions = []
        options = {}
        columns_to_drop = []
        replace_value = ""
        replace_with = ""
        replace_scope = "All columns"
        encode_cols = []
        encode_method = "Label Encoding"
        enrich_col = "None"
        enrich_api_key = ""
        train_ml = False
        target_col = None
        feature_cols = []
        custom_rules = []

        with manual_container:
            st.subheader("Manual Column Dropping")
            st.markdown('<span title="Select columns to remove from the dataset">ℹ️</span>', unsafe_allow_html=True)
            columns_to_drop = st.multiselect(
                "Select columns to drop", 
                available_columns, 
                help="Choose columns to remove from the dataset. These columns will not appear in subsequent steps."
            )
        
        with custom_rules_container:
            with st.expander("Custom Cleaning Rules", expanded=False):
                st.markdown("**Define custom cleaning rules**")
                st.markdown('<span title="Create rules like \'if column X > 100, set to NaN\' to apply custom transformations">ℹ️</span>', unsafe_allow_html=True)
                num_rules = st.number_input("Number of Custom Rules", min_value=0, max_value=10, value=0, step=1)
                for i in range(num_rules):
                    with st.container():
                        st.write(f"**Rule {i+1}**")
                        rule_col = st.selectbox(f"Select column for Rule {i+1}", available_columns, key=f"rule_col_{i}")
                        condition = st.selectbox(f"Condition for Rule {i+1}", ["greater than", "less than", "equal to"], key=f"rule_cond_{i}")
                        threshold = st.number_input(f"Threshold for Rule {i+1}", value=0.0, key=f"rule_threshold_{i}")
                        action = st.selectbox(f"Action for Rule {i+1}", ["Set to NaN", "Set to Value"], key=f"rule_action_{i}")
                        if action == "Set to Value":
                            action_value = st.number_input(f"Set Value for Rule {i+1}", value=0.0, key=f"rule_action_value_{i}")
                        else:
                            action_value = None
                        custom_rules.append({
                            "column": rule_col,
                            "condition": condition,
                            "threshold": threshold,
                            "action": action,
                            "action_value": action_value
                        })
        
        with replace_container:
            with st.expander("Custom Value Replacement", expanded=False):
                st.markdown("**Replace unwanted values (e.g., '?' for missing data)**")
                st.markdown('<span title="Replace specific values across selected columns (e.g., \'?\' with NaN)">ℹ️</span>', unsafe_allow_html=True)
                replace_value = st.text_input(
                    "Value to replace (e.g., ?, 999, Unknown)", 
                    "", 
                    help="Enter the value you want to replace. Case-insensitive matching will be applied."
                )
                replace_with = st.radio(
                    "Replace with", 
                    ["NaN", "?", "0", "Custom"], 
                    help="Select what to replace the value with. 'NaN' is recommended for missing data."
                )
                if replace_with == "Custom":
                    replace_with = st.text_input(
                        "Custom replacement value", 
                        "", 
                        help="Enter a custom replacement value."
                    )
                replace_scope = st.radio(
                    "Apply to", 
                    ["All columns", "Numeric columns", "Categorical columns"], 
                    help="Choose which columns to apply the replacement to."
                )
        
        with encode_container:
            with st.expander("Convert Categorical to Numerical", expanded=False):
                st.markdown('<span title="Convert categorical columns to numerical values for ML compatibility">ℹ️</span>', unsafe_allow_html=True)
                cat_cols = [col for col in df[available_columns].select_dtypes(include=['object', 'category']).columns.tolist() if col in available_columns]
                encode_cols = st.multiselect(
                    "Select categorical columns to convert", 
                    cat_cols, 
                    help="Choose categorical columns to convert to numerical."
                )
                encode_method = st.radio(
                    "Conversion method", 
                    ["Label Encoding", "One-Hot Encoding"], 
                    help="Label Encoding assigns integers; One-Hot creates dummy columns."
                )
        
        with enrich_container:
            with st.expander("Smart Data Enrichment", expanded=False):
                st.markdown('<span title="Enrich data with external information (e.g., geolocation from addresses)">ℹ️</span>', unsafe_allow_html=True)
                enrich_col = st.selectbox(
                    "Column to Enrich (e.g., address)", 
                    ["None"] + available_columns, 
                    help="Select a column to enrich with external data."
                )
                enrich_api_key = st.text_input(
                    "Google API Key (for geolocation)", 
                    type="password", 
                    help="Enter your Google Maps API key."
                )
                if enrich_col != "None" and not enrich_api_key:
                    st.warning("Google API Key is required for data enrichment.")
        
        with ai_container:
            with st.expander("AI Cleaning Suggestions", expanded=True):
                st.markdown('<span title="AI-driven suggestions to automate data cleaning (e.g., fill missing values, remove duplicates)">ℹ️</span>', unsafe_allow_html=True)
                for suggestion, explanation in st.session_state.suggestions:
                    if "Based on the provided dataset analysis" in suggestion:
                        st.markdown(f"**{suggestion}** - {explanation}")
                    else:
                        if st.checkbox(f"{suggestion}", key=suggestion):
                            selected_suggestions.append((suggestion, explanation))
                            st.session_state.ai_suggestions_used += 1
                            st.markdown(f"**Explanation:** {explanation}")
                            if "Handle special characters" in suggestion:
                                options["special_chars"] = st.radio(
                                    "Action for special characters", 
                                    ("Drop them", "Replace with underscores"), 
                                    key=f"special_chars_opt_{suggestion}"
                                )
                            elif "Fill missing values" in suggestion:
                                col = extract_column(suggestion)
                                if col and col in available_columns and df[col].dtype in ['int64', 'float64']:
                                    options[f"fill_{col}"] = st.radio(
                                        f"Fill method for {col}", 
                                        ["mean", "median", "mode"], 
                                        key=f"fill_opt_{col}_{suggestion}"
                                    )
                            elif "Handle outliers" in suggestion:
                                col = extract_column(suggestion)
                                if col and col in available_columns:
                                    options[f"outlier_{col}"] = st.radio(
                                        f"Action for outliers in {col}", 
                                        ("Remove", "Cap at bounds"), 
                                        key=f"outlier_opt_{col}_{suggestion}"
                                    )
        
        with anomaly_container:
            with st.expander("Anomaly Detection", expanded=False):
                st.markdown('<span title="Detect outliers in numerical columns using Isolation Forest algorithm">ℹ️</span>', unsafe_allow_html=True)
                num_cols = [col for col in df[available_columns].select_dtypes(include=['int64', 'float64']).columns.tolist() if col in available_columns]
                anomaly_cols = st.multiselect(
                    "Select numerical columns for anomaly detection", 
                    num_cols, 
                    help="Detect outliers using AI."
                )
                contamination = st.slider(
                    "Contamination factor", 
                    0.01, 0.5, 0.1, 
                    help="Percentage of data expected to be anomalies."
                )
                if anomaly_cols:
                    with st.spinner("Detecting anomalies..."):
                        try:
                            anomalies = detect_anomalies(df[available_columns], anomaly_cols, contamination)
                            st.write("Anomalies Detected:")
                            st.json(anomalies)
                        except Exception as e:
                            st.error(f"Error detecting anomalies: {str(e)}. Please ensure the selected columns contain valid numerical data.")

        with ml_container:
            with st.expander("One-Click ML Deployment", expanded=False):
                st.markdown('<span title="Train a machine learning model and deploy it as a Streamlit app with a single click">ℹ️</span>', unsafe_allow_html=True)
                target_col = st.selectbox(
                    "Target Column (to predict)", 
                    available_columns, 
                    help="Column to predict with ML."
                )
                feature_cols = st.multiselect(
                    "Feature Columns", 
                    [col for col in available_columns if col != target_col], 
                    help="Columns to use as predictors."
                )
                train_ml = st.checkbox("Train and Deploy ML Model", help="Generate a prediction app.")
                if train_ml and not (target_col and feature_cols):
                    st.warning("Please select a target column and at least one feature column for ML deployment.")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            preview_button = st.form_submit_button(label="Preview Changes")
        with col2:
            apply_button = st.form_submit_button(label="Apply Changes")
        with col3:
            auto_clean_button = st.form_submit_button(label="Auto-Clean")

    if preview_button or apply_button or auto_clean_button:
        operations_selected = (
            selected_suggestions or
            columns_to_drop or
            (replace_value.strip() and replace_with) or
            encode_cols or
            (enrich_col != "None" and enrich_api_key) or
            auto_clean_button or
            (train_ml and target_col and feature_cols) or
            custom_rules
        )
        if not operations_selected:
            st.warning("Please select at least one cleaning operation, custom rule, or ML deployment with valid parameters.")
        else:
            with st.spinner("Processing cleaning operations..."):
                try:
                    # Validate inputs for custom value replacement
                    if replace_value.strip() and replace_with:
                        if replace_with == "Custom" and not replace_with.strip():
                            st.error("Please provide a custom replacement value.")
                            return
                        if replace_scope not in ["All columns", "Numeric columns", "Categorical columns"]:
                            st.error("Invalid replacement scope selected.")
                            return

                    cleaned_df, logs = apply_cleaning_operations(
                        df, selected_suggestions, columns_to_drop, options, 
                        replace_value, replace_with if replace_with != "NaN" else "NaN", 
                        replace_scope, encode_cols, encode_method, auto_clean=auto_clean_button, 
                        enrich_col=enrich_col if enrich_col != "None" else None, enrich_api_key=enrich_api_key,
                        train_ml=train_ml, target_col=target_col, feature_cols=feature_cols
                    )

                    # Apply custom cleaning rules
                    for rule in custom_rules:
                        col = rule["column"]
                        condition = rule["condition"]
                        threshold = rule["threshold"]
                        action = rule["action"]
                        action_value = rule["action_value"]
                        
                        if col in cleaned_df.columns:
                            if condition == "greater than":
                                mask = cleaned_df[col] > threshold
                            elif condition == "less than":
                                mask = cleaned_df[col] < threshold
                            else:  # equal to
                                mask = cleaned_df[col] == threshold
                            
                            if action == "Set to NaN":
                                cleaned_df.loc[mask, col] = np.nan
                            else:
                                cleaned_df.loc[mask, col] = action_value
                            
                            logs.append(f"Applied custom rule on {col}: {condition} {threshold}, {action} {'NaN' if action == 'Set to NaN' else action_value}")

                    if preview_button:
                        st.subheader("Preview of Changes")
                        st.write("Before:")
                        st.dataframe(df.head(10))
                        st.write("After:")
                        st.dataframe(cleaned_df.head(10))
                        st.write("Preview Logs:")
                        for log in logs:
                            st.write(f"- {log}")
                
                    if apply_button or auto_clean_button:
                        st.session_state.previous_states.append((df.copy(), st.session_state.logs.copy()))
                        if len(st.session_state.previous_states) > 5:
                            st.session_state.previous_states.pop(0)
                        st.session_state.redo_states = []
                        st.session_state.cleaned_df = cleaned_df
                        st.session_state.logs = logs
                        if columns_to_drop:
                            st.session_state.dropped_columns.extend(columns_to_drop)
                        st.session_state.cleaning_history.append({
                            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "logs": logs
                        })
                        st.session_state.suggestions = get_cached_suggestions(cleaned_df[[col for col in cleaned_df.columns if col not in st.session_state.dropped_columns]])
                        st.success("Changes applied successfully!")
                        st.session_state.progress["Clean"] = "Done"
                        # Collapse the Custom Value Replacement section after applying changes
                        st.session_state['replace_expanded'] = False
                        st.rerun()
                except Exception as e:
                    st.error(f"Error processing cleaning operations: {str(e)}. Please check your inputs and try again.")
                    st.session_state.progress["Clean"] = "Failed"

    with st.expander("Save/Apply Cleaning Templates", expanded=False):
        st.subheader("Save/Apply Cleaning Templates")
        st.markdown('<span title="Save your cleaning configuration as a template to reuse later, or apply a saved template">ℹ️</span>', unsafe_allow_html=True)
        with st.form(key="template_form"):
            template_name = st.text_input("Template Name", "", help="Enter a name to save this cleaning configuration.")
            save_template_button = st.form_submit_button("Save as Template")

            if save_template_button and template_name:
                template = {
                    "columns_to_drop": columns_to_drop,
                    "selected_suggestions": selected_suggestions,
                    "options": options,
                    "replace_value": replace_value,
                    "replace_with": replace_with,
                    "replace_scope": replace_scope,
                    "encode_cols": encode_cols,
                    "encode_method": encode_method,
                    "enrich_col": enrich_col,
                    "train_ml": train_ml,
                    "target_col": target_col,
                    "feature_cols": feature_cols,
                    "custom_rules": custom_rules
                }
                st.session_state.cleaning_templates[template_name] = {k: v for k, v in template.items() if k != "enrich_api_key"}
                st.success(f"Saved template '{template_name}'")

        if st.session_state.cleaning_templates:
            with st.form(key="apply_template_form"):
                template_to_apply = st.selectbox("Apply Saved Template", ["None"] + list(st.session_state.cleaning_templates.keys()))
                apply_template_button = st.form_submit_button("Apply Template")

                if apply_template_button and template_to_apply != "None":
                    template = st.session_state.cleaning_templates[template_to_apply]
                    with st.spinner("Applying template..."):
                        try:
                            cleaned_df, logs = apply_cleaning_operations(
                                df, 
                                selected_suggestions=template["selected_suggestions"],
                                columns_to_drop=template["columns_to_drop"],
                                options=template["options"],
                                replace_value=template["replace_value"],
                                replace_with=template["replace_with"],
                                replace_scope=template["replace_scope"],
                                encode_cols=template["encode_cols"],
                                encode_method=template["encode_method"],
                                auto_clean=False,
                                enrich_col=template["enrich_col"],
                                enrich_api_key=enrich_api_key,
                                train_ml=template["train_ml"],
                                target_col=template["target_col"],
                                feature_cols=template["feature_cols"]
                            )
                            st.session_state.previous_states.append((df.copy(), st.session_state.logs.copy()))
                            if len(st.session_state.previous_states) > 5:
                                st.session_state.previous_states.pop(0)
                            st.session_state.redo_states = []
                            st.session_state.cleaned_df = cleaned_df
                            st.session_state.logs = logs
                            if template["columns_to_drop"]:
                                st.session_state.dropped_columns.extend(template["columns_to_drop"])
                            st.session_state.cleaning_history.append({
                                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                "logs": logs + [f"Applied template '{template_to_apply}'"]
                            })
                            st.session_state.suggestions = get_cached_suggestions(cleaned_df[[col for col in cleaned_df.columns if col not in st.session_state.dropped_columns]])
                            st.success(f"Applied template '{template_to_apply}'")
                            st.session_state.progress["Clean"] = "Done"
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error applying template: {str(e)}. Please check the template and try again.")
                            st.session_state.progress["Clean"] = "Failed"

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.previous_states and st.button("Undo Last Cleaning", help="Revert to the previous state"):
            current_state = (st.session_state.cleaned_df.copy(), st.session_state.logs.copy())
            st.session_state.redo_states.append(current_state)
            if len(st.session_state.redo_states) > 5:
                st.session_state.redo_states.pop(0)
            previous_df, previous_logs = st.session_state.previous_states.pop()
            st.session_state.cleaned_df = previous_df
            st.session_state.logs = previous_logs
            st.session_state.suggestions = get_cached_suggestions(previous_df[[col for col in previous_df.columns if col not in st.session_state.dropped_columns]])
            st.session_state.cleaning_history.append({
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "logs": ["Undid last cleaning operation"]
            })
            st.rerun()

    with col2:
        if st.session_state.redo_states and st.button("Redo Last Cleaning", help="Reapply the last undone state"):
            st.session_state.previous_states.append((st.session_state.cleaned_df.copy(), st.session_state.logs.copy()))
            if len(st.session_state.previous_states) > 5:
                st.session_state.previous_states.pop(0)
            redo_df, redo_logs = st.session_state.redo_states.pop()
            st.session_state.cleaned_df = redo_df
            st.session_state.logs = redo_logs
            st.session_state.suggestions = get_cached_suggestions(redo_df[[col for col in redo_df.columns if col not in st.session_state.dropped_columns]])
            st.session_state.cleaning_history.append({
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "logs": ["Redid last cleaning operation"]
            })
            st.rerun()

    with st.expander("Cleaning History", expanded=False):
        st.subheader("Cleaning History")
        if st.session_state.cleaning_history:
            for entry in st.session_state.cleaning_history:
                st.write(f"**{entry['timestamp']}**")
                for log in entry['logs']:
                    st.write(f"- {log}")
        else:
            st.write("No cleaning operations have been performed yet.")

    with st.expander("Export to Tableau", expanded=False):
        st.subheader("Export to Tableau")
        st.markdown('<span title="Export your cleaned dataset as a CSV file for use in Tableau">ℹ️</span>', unsafe_allow_html=True)
        if st.session_state.cleaned_df is not None:
            export_button = st.button("Export Cleaned Dataset for Tableau")
            if export_button:
                filename = f"cleaned_for_tableau_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                st.markdown(get_download_link(st.session_state.cleaned_df, filename), unsafe_allow_html=True)
                st.info("Download the CSV and import it into Tableau Public or Desktop to create visualizations!")

    if st.session_state.cleaned_df is not None:
        st.subheader("Cleaned Dataset Preview")
        view_option = st.radio("View dataset as:", ("First 10 Rows", "Full Dataset"), horizontal=True)
        if view_option == "First 10 Rows":
            st.dataframe(st.session_state.cleaned_df.head(10))
        else:
            # Display full dataset with scrollable view (pagination already removed)
            st.dataframe(st.session_state.cleaned_df, use_container_width=True, height=600)
        
        st.subheader("Cleaning Summary")
        st.write(f"Original Shape: {st.session_state.df.shape}")
        st.write(f"New Shape: {st.session_state.cleaned_df.shape}")
        st.write(f"New Health Score: {calculate_health_score(st.session_state.cleaned_df)}/100")
        for log in st.session_state.logs:
            st.write(f"- {log}")
        st.markdown(get_download_link(st.session_state.cleaned_df, 
                                    f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), 
                   unsafe_allow_html=True)

def render_insights_page() -> None:
    """
    Render the insights page with natural language generation.
    """
    st.title("Insights Dashboard")
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("Please upload a dataset first on the Upload page.")
        return

    # Update progress
    st.session_state.progress["Insights"] = "In Progress"

    df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df
    available_columns = [col for col in df.columns if col not in st.session_state.dropped_columns]

    with st.spinner("Generating insights..."):
        try:
            insights = get_insights(df[available_columns])
            st.subheader("Key Insights")
            for insight in insights:
                st.write(f"- {insight}")
            st.session_state.progress["Insights"] = "Done"
        except Exception as e:
            st.error(f"Error generating insights: {str(e)}. Please ensure the dataset is valid and try again.")
            st.session_state.progress["Insights"] = "Failed"

def render_predictive_page(df: pd.DataFrame) -> None:
    """
    Render the predictive analytics page with ML model training, forecasting, and clustering.
    """
    st.title("Predictive Analytics")
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("Please upload a dataset first on the Upload page.")
        return

    # Update progress
    st.session_state.progress["Predictive"] = "In Progress"

    available_columns = [col for col in df.columns if col not in st.session_state.dropped_columns]
    df = df[available_columns]

    st.subheader("Predictive Dashboard")
    render_predictive_page_external(df)

    st.subheader("Generate Synthetic Data")
    task_type = st.selectbox("Task Type", ["classification", "regression"])
    if st.button("Generate Synthetic Data"):
        with st.spinner("Generating synthetic data..."):
            try:
                synthetic_df = generate_synthetic_data(df, task_type)
                st.session_state.cleaned_df = synthetic_df
                st.session_state.suggestions = get_cached_suggestions(synthetic_df[[col for col in synthetic_df.columns if col not in st.session_state.dropped_columns]])
                st.session_state.cleaning_history.append({
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "logs": ["Generated synthetic data"]
                })
                st.write("Synthetic Dataset Preview:")
                st.dataframe(synthetic_df.head(10))
                st.markdown(get_download_link(synthetic_df, 
                                            f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), 
                           unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error generating synthetic data: {str(e)}. Please check the dataset and try again.")
                st.session_state.progress["Predictive"] = "Failed"

    st.subheader("Time Series Forecasting")
    time_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    if time_cols:
        forecast_col = st.selectbox("Select time series column", time_cols)
        periods = st.slider("Forecast periods", 1, 30, 5)
        freq = st.selectbox("Frequency", ["D", "M", "Y"], help="Select the frequency of the time series data.")
        if st.button("Forecast"):
            with st.spinner("Forecasting..."):
                try:
                    forecast_df = forecast_time_series(df, forecast_col, periods, time_col=forecast_col, freq=freq)
                    st.write("Forecasted Values:")
                    st.dataframe(forecast_df)
                    st.markdown(get_download_link(forecast_df, 
                                                f"forecast_{forecast_col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), 
                               unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error forecasting time series: {str(e)}. Please ensure the time column is in datetime format.")
                    st.session_state.progress["Predictive"] = "Failed"
    else:
        st.info("No datetime columns found for time series forecasting.")

    st.subheader("Time Series Decomposition")
    if time_cols:
        decompose_col = st.selectbox("Select column for decomposition", time_cols, key="decompose_col")
        period = st.slider("Period for decomposition", 1, 30, 12)
        if st.button("Decompose Time Series"):
            with st.spinner("Decomposing time series..."):
                try:
                    decomposition = analyze_time_series(df, decompose_col, period)
                    if decomposition:
                        st.write("Trend Component:")
                        st.line_chart(decomposition.get("trend"))
                        st.write("Seasonal Component:")
                        st.line_chart(decomposition.get("seasonal"))
                        st.write("Residual Component:")
                        st.line_chart(decomposition.get("residual"))
                    else:
                        st.error("Failed to decompose time series. Ensure the column has sufficient data.")
                        st.session_state.progress["Predictive"] = "Failed"
                except Exception as e:
                    st.error(f"Error decomposing time series: {str(e)}. Please check the selected column.")
                    st.session_state.progress["Predictive"] = "Failed"
    else:
        st.info("No datetime columns found for time series decomposition.")

    st.subheader("Clustering")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cluster_cols = st.multiselect("Select columns for clustering", numeric_cols)
    n_clusters = st.slider("Number of clusters", 2, 10, 3)
    if st.button("Perform Clustering", key="ui_perform_clustering"):
        if len(cluster_cols) < 2:
            st.warning("Please select at least two columns for clustering.")
        else:
            with st.spinner("Performing clustering..."):
                try:
                    labels = perform_clustering(df, cluster_cols, n_clusters)
                    df['Cluster'] = labels
                    st.session_state.cleaned_df = df
                    st.session_state.suggestions = get_cached_suggestions(df[[col for col in df.columns if col not in st.session_state.dropped_columns]])
                    st.session_state.cleaning_history.append({
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "logs": ["Performed clustering"]
                    })
                    st.write("Dataset with Cluster Labels:")
                    st.dataframe(df.head(10))
                    st.markdown(get_download_link(df, 
                                                f"clustered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), 
                               unsafe_allow_html=True)
                    st.session_state.progress["Predictive"] = "Done"
                except Exception as e:
                    st.error(f"Error performing clustering: {str(e)}. Please ensure the selected columns contain valid numerical data.")
                    st.session_state.progress["Predictive"] = "Failed"