import base64
import json
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq  # For Parquet file support
import streamlit as st

from data_utils import (analyze_time_series, apply_cleaning_operations,
                        calculate_health_score, chat_with_gpt,
                        detect_anomalies, extract_column, forecast_time_series,
                        generate_synthetic_data, get_cleaning_suggestions,
                        get_insights, perform_clustering, suggest_workflow,
                        train_ml_model)
from predictive import \
    render_predictive_page as render_predictive_page_external

# Set up logging with rotation
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Change to INFO for production
if not logger.handlers:  # Avoid adding handlers multiple times
    handler = RotatingFileHandler(
        'ui.log',
        maxBytes=5 * 1024 * 1024,
        backupCount=3)  # 5MB per file, keep 3 backups
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# Cache expensive operations


@st.cache_data
def get_cached_suggestions(df: pd.DataFrame) -> List[Tuple[str, str]]:
    return get_cleaning_suggestions(df)


def get_download_link(df: pd.DataFrame, filename: str) -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'


def profile_dataset(df: pd.DataFrame) -> Dict[str, any]:
    profile = {}
    for col in df.columns:
        col_profile = {}
        col_types = df[col].apply(type).nunique()
        col_profile['mixed_types'] = col_types > 1
        col_profile['type_suggestion'] = f"Convert {col} to {
            df[col].dtype.name}" if col_types > 1 else None

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            formats = df[col].dropna().apply(
                lambda x: x.strftime('%Y-%m-%d')).nunique()
            col_profile['inconsistent_formats'] = formats > 1
            col_profile['format_suggestion'] = "Standardize date format to YYYY-MM-DD" if formats > 1 else None

        missing_percentage = df[col].isna().mean() * 100
        col_profile['missing_percentage'] = missing_percentage
        col_profile['missing_suggestion'] = f"Consider filling or dropping {col} (missing {
            missing_percentage:.2f}%)" if missing_percentage > 10 else None

        profile[col] = col_profile
    return profile


def initialize_session_state() -> None:
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
        },
        'cleaned_view_option': "First 10 Rows"  # New: Persist view option
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def display_cleaned_dataset(cleaned_df: pd.DataFrame) -> None:
    if cleaned_df is None or cleaned_df.empty:
        st.warning("No cleaned dataset available to display.")
        return

    if st.session_state.cleaned_df is not None:
        st.subheader("Cleaned Dataset Preview")
        view_option = st.radio(
            "View dataset as:",
            ("First 10 Rows", "Full Dataset"),
            horizontal=True)
        if view_option == "First 10 Rows":
            st.dataframe(st.session_state.cleaned_df.head(10))
        else:
            st.dataframe(st.session_state.cleaned_df, use_container_width=True, height=600)

    try:
        st.write(f"Dataset size: {cleaned_df.shape}")
        if view_option == "First 10 Rows":
            st.dataframe(cleaned_df.head(10), use_container_width=True)
        else:
            if len(cleaned_df) > 1000:
                st.warning(f"Dataset has {len(cleaned_df)} rows. Displaying first 1000 rows to avoid performance issues.")
                st.dataframe(cleaned_df.head(1000), use_container_width=True)
            else:
                st.dataframe(cleaned_df, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying dataset: {str(e)}")

    # Summary (always visible)
    st.subheader("Cleaning Summary")
    st.write(f"Original Shape: {st.session_state.df.shape}")
    st.write(f"New Shape: {cleaned_df.shape}")
    st.write(f"New Health Score: {calculate_health_score(cleaned_df)}/100")
    for log in st.session_state.logs:
        st.write(f"- {log}")
    st.markdown(
        get_download_link(
            cleaned_df,
            f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
        unsafe_allow_html=True)

def render_upload_page() -> None:
    st.title("Upload Your Dataset")
    st.markdown(
        "<p class='welcome'>Start your data journey here!</p>",
        unsafe_allow_html=True)

    initialize_session_state()
    st.session_state.progress["Upload"] = "In Progress"

    # Always display the upload widget
    uploaded_file = st.file_uploader(
        "Choose a file (CSV, Excel, JSON, or Parquet)",
        type=["csv", "xlsx", "json", "parquet"],
        help="Upload a dataset file to begin.",
        key="file_uploader"
    )

    # Display metadata if dataset exists (no buttons)
    if st.session_state.df is not None:
        st.subheader("Original Dataset Preview (First 10 Rows)")
        # Updated with use_container_width
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        st.subheader("Basic Metadata")
        score = calculate_health_score(st.session_state.df)
        st.write(f"Rows: {st.session_state.df.shape[0]}")
        st.write(f"Columns: {st.session_state.df.shape[1]}")
        st.write(f"Missing Values: {st.session_state.df.isna().sum().sum()}")
        st.progress(score / 100)
        st.write(f"Dataset Health Score: {score}/100")
        st.info(
            "This is the original dataset. Cleaning operations are applied to a working copy.")
        st.warning(
            "Uploading a new file will overwrite the current dataset and reset all cleaning operations. Proceed with caution!")

    # Handle file upload
    if uploaded_file:
        try:
            with st.spinner("Loading dataset..."):
                if uploaded_file.size > 50 * 1024 * 1024:  # 50MB
                    st.warning(
                        "File size exceeds 50MB. Using chunked processing.")
                    if uploaded_file.name.endswith('.csv'):
                        chunks = pd.read_csv(uploaded_file, chunksize=10000)
                        df_list = []
                        progress_bar = st.progress(0)
                        total_chunks = uploaded_file.size // (10000 * 100) or 1
                        for i, chunk in enumerate(chunks):
                            df_list.append(chunk)
                            progress_bar.progress(
                                min((i + 1) / total_chunks, 1.0))
                        df = pd.concat(df_list, ignore_index=True)
                    elif uploaded_file.name.endswith('.json'):
                        chunks = pd.read_json(uploaded_file, chunksize=10000)
                        df_list = []
                        progress_bar = st.progress(0)
                        total_chunks = uploaded_file.size // (10000 * 100) or 1
                        for i, chunk in enumerate(chunks):
                            df_list.append(chunk)
                            progress_bar.progress(
                                min((i + 1) / total_chunks, 1.0))
                        df = pd.concat(df_list, ignore_index=True)
                    elif uploaded_file.name.endswith('.parquet'):
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

                if df.shape[0] > 4000:
                    st.info(
                        f"Large dataset detected ({
                            df.shape[0]} rows). Processing optimized for performance.")
                if df.empty:
                    st.error(
                        "Uploaded dataset is empty. Please upload a valid file.")
                    return

                with st.spinner("Profiling dataset..."):
                    profile = profile_dataset(df)
                    st.subheader("Dataset Profile")
                    for col, info in profile.items():
                        if any(info.values()):
                            st.write(f"**Column: {col}**")
                            if info['mixed_types']:
                                st.write(
                                    f"- Mixed Types Detected: {info['mixed_types']}")
                                st.write(
                                    f"  Suggestion: {
                                        info['type_suggestion']}")
                            if info.get('inconsistent_formats'):
                                st.write(
                                    f"- Inconsistent Formats: {info['inconsistent_formats']}")
                                st.write(
                                    f"  Suggestion: {
                                        info['format_suggestion']}")
                            if info['missing_percentage'] > 10:
                                st.write(
                                    f"- Missing Values: {info['missing_percentage']:.2f}%")
                                st.write(
                                    f"  Suggestion: {
                                        info['missing_suggestion']}")

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

                st.success("Dataset uploaded successfully!")
                st.session_state.progress["Upload"] = "Done"
                from app import save_auth_state
                save_auth_state()
                st.rerun()  # Rerun to refresh the UI after upload
        except Exception as e:
            st.error(
                f"Error loading file: {
                    str(e)}. Please ensure the file is a valid CSV, Excel, JSON, or Parquet file.")
            st.session_state.progress["Upload"] = "Failed"


def render_clean_page() -> None:
    st.title("Clean Your Dataset")
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("Please upload a dataset first on the Upload page.")
        return

    df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df
    available_columns = [
        col for col in df.columns if col not in st.session_state.dropped_columns]

    if not available_columns:
        st.error("No columns available for cleaning. Please upload a new dataset.")
        return

    st.session_state.progress["Clean"] = "In Progress"

    st.subheader("Your Progress")
    progress_text = ""
    for step, status in st.session_state.progress.items():
        emoji = "✅" if status == "Done" else "🟡" if status == "In Progress" else "⬜"
        progress_text += f"{emoji} {step}: {status}\n"
    st.markdown(progress_text)

    if not st.session_state.suggestions or id(
            st.session_state.cleaned_df) != id(df):
        with st.spinner("Generating AI cleaning suggestions..."):
            st.session_state.suggestions = get_cached_suggestions(
                df[available_columns])

    st.subheader("Dataset Health")
    score = calculate_health_score(df)
    st.progress(score / 100)
    st.write(f"Current Health Score: {score}/100")

    st.subheader("Smart Workflow Automation")
    st.markdown(
        '<span title="Run an AI-suggested cleaning workflow automatically">ℹ️</span>',
        unsafe_allow_html=True)
    if st.button("Run Smart Workflow", key="run_smart_workflow_button"):
        logger.info("Run Smart Workflow button clicked")
        with st.spinner("Generating and executing workflow..."):
            try:
                workflow = suggest_workflow(df[available_columns])
                st.write("### Suggested Workflow:")
                for step in workflow:
                    st.write(f"- {step}")
                cleaned_df, logs = apply_cleaning_operations(
                    df, st.session_state.suggestions, [], {}, "", "", "", [], "", auto_clean=True
                )
                st.session_state.previous_states.append(
                    (df.copy(), st.session_state.logs.copy()))
                st.session_state.cleaned_df = cleaned_df
                st.session_state.logs = logs
                st.session_state.redo_states = []
                st.session_state.cleaning_history.append({
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "logs": logs + ["Executed Smart Workflow"]
                })
                st.session_state.suggestions = get_cached_suggestions(
                    cleaned_df[[col for col in cleaned_df.columns if col not in st.session_state.dropped_columns]])
                st.success("Smart Workflow executed successfully!")
                st.session_state.progress["Clean"] = "Done"
                display_cleaned_dataset(st.session_state.cleaned_df)
                from app import save_auth_state
                save_auth_state()
            except Exception as e:
                st.error(f"Error executing smart workflow: {str(e)}")
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
            st.markdown(
                '<span title="Select columns to remove from the dataset">ℹ️</span>',
                unsafe_allow_html=True)
            columns_to_drop = st.multiselect(
                "Select columns to drop",
                available_columns,
                help="Choose columns to remove from the dataset",
                key="columns_to_drop"
            )

        with custom_rules_container:
            with st.expander("Custom Cleaning Rules", expanded=False):
                st.markdown("**Define custom cleaning rules**")
                st.markdown(
                    '<span title="Create rules like \'if column X > 100, set to NaN\'">ℹ️</span>',
                    unsafe_allow_html=True)
                num_rules = st.number_input(
                    "Number of Custom Rules",
                    min_value=0,
                    max_value=10,
                    value=0,
                    step=1,
                    key="num_rules")
                for i in range(num_rules):
                    with st.container():
                        st.write(f"**Rule {i + 1}**")
                        rule_col = st.selectbox(
                            f"Select column for Rule {
                                i + 1}", available_columns, key=f"rule_col_{i}")
                        condition = st.selectbox(
                            f"Condition for Rule {
                                i + 1}",
                            [
                                "greater than",
                                "less than",
                                "equal to"],
                            key=f"rule_cond_{i}")
                        threshold = st.number_input(
                            f"Threshold for Rule {
                                i + 1}", value=0.0, key=f"rule_threshold_{i}")
                        action = st.selectbox(
                            f"Action for Rule {i + 1}", ["Set to NaN", "Set to Value"], key=f"rule_action_{i}")
                        if action == "Set to Value":
                            action_value = st.number_input(
                                f"Set Value for Rule {
                                    i + 1}", value=0.0, key=f"rule_action_value_{i}")
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
                st.markdown(
                    "**Replace unwanted values (e.g., '?' for missing data)**")
                st.markdown(
                    '<span title="Replace specific values across selected columns">ℹ️</span>',
                    unsafe_allow_html=True)
                replace_value = st.text_input(
                    "Value to Replace",  # Non-empty label
                    value="",
                    help="Enter the value you want to replace (e.g., ?, 999, Unknown)",
                    key="replace_value"
                )
                replace_with = st.radio(
                    "Replace with",
                    ["NaN", "?", "0", "Custom"],
                    help="Select what to replace the value with",
                    key="replace_with"
                )
                if replace_with == "Custom":
                    replace_with = st.text_input(
                        "Custom replacement value",
                        "",
                        help="Enter a custom replacement value",
                        key="replace_with_custom"
                    )
                replace_scope = st.radio(
                    "Apply to",
                    ["All columns", "Numeric columns", "Categorical columns"],
                    help="Choose which columns to apply the replacement to",
                    key="replace_scope"
                )

        with encode_container:
            with st.expander("Convert Categorical to Numerical", expanded=False):
                st.markdown(
                    '<span title="Convert categorical columns to numerical values">ℹ️</span>',
                    unsafe_allow_html=True)
                cat_cols = [col for col in df[available_columns].select_dtypes(
                    include=['object', 'category']).columns.tolist() if col in available_columns]
                encode_cols = st.multiselect(
                    "Select categorical columns to convert",
                    cat_cols,
                    help="Choose categorical columns to convert to numerical",
                    key="encode_cols"
                )
                encode_method = st.radio(
                    "Conversion method",
                    ["Label Encoding", "One-Hot Encoding"],
                    help="Label Encoding assigns integers; One-Hot creates dummy columns",
                    key="encode_method"
                )

        with enrich_container:
            with st.expander("Smart Data Enrichment", expanded=False):
                st.markdown(
                    '<span title="Enrich data with external info (e.g., geolocation)">ℹ️</span>',
                    unsafe_allow_html=True)
                enrich_col = st.selectbox(
                    "Column to Enrich (e.g., address)",
                    ["None"] + available_columns,
                    help="Select a column to enrich with external data",
                    key="enrich_col"
                )
                enrich_api_key = st.text_input(
                    "Google API Key (for geolocation)",
                    type="password",
                    help="Enter your Google Maps API key",
                    key="enrich_api_key"
                )
                if enrich_col != "None" and not enrich_api_key:
                    st.warning(
                        "Google API Key is required for data enrichment.")

        with ai_container:
            with st.expander("AI Cleaning Suggestions", expanded=True):
                st.markdown(
                    '<span title="AI-driven suggestions to automate data cleaning">ℹ️</span>',
                    unsafe_allow_html=True)
                for idx, (suggestion, explanation) in enumerate(
                        st.session_state.suggestions):
                    if "Based on the provided dataset analysis" in suggestion:
                        st.markdown(f"{suggestion} - {explanation}")
                    else:
                        if st.checkbox(f"{suggestion}",
                                       key=f"suggestion_{suggestion}_{idx}"):
                            selected_suggestions.append(
                                (suggestion, explanation))
                            st.session_state.ai_suggestions_used += 1
                            st.markdown(f"Explanation: {explanation}")
                            if "Handle special characters" in suggestion:
                                options["special_chars"] = st.radio(
                                    "Action for special characters",
                                    ("Drop them", "Replace with underscores"),
                                    key=f"special_chars_opt_{suggestion}_{idx}"
                                )
                            elif "Fill missing values" in suggestion:
                                col = extract_column(suggestion)
                                if col and col in available_columns and df[col].dtype in [
                                        'int64', 'float64']:
                                    options[f"fill_{col}"] = st.radio(
                                        f"Fill method for {col}",
                                        ["mean", "median", "mode"],
                                        key=f"fill_opt_{col}_{suggestion}_{idx}"
                                    )
                            elif "Handle outliers" in suggestion:
                                col = extract_column(suggestion)
                                if col and col in available_columns:
                                    options[f"outlier_{col}"] = st.radio(
                                        f"Action for outliers in {col}",
                                        ("Remove", "Cap at bounds"),
                                        key=f"outlier_opt_{col}_{suggestion}_{idx}"
                                    )

        with anomaly_container:
            with st.expander("Anomaly Detection", expanded=False):
                st.markdown(
                    '<span title="Detect outliers in numerical columns">ℹ️</span>',
                    unsafe_allow_html=True)
                num_cols = [col for col in df[available_columns].select_dtypes(
                    include=['int64', 'float64']).columns.tolist() if col in available_columns]
                anomaly_cols = st.multiselect(
                    "Select numerical columns for anomaly detection",
                    num_cols,
                    help="Detect outliers using AI",
                    key="anomaly_cols"
                )
                contamination = st.slider(
                    "Contamination factor",
                    0.01, 0.5, 0.1,
                    help="Percentage of data expected to be anomalies",
                    key="contamination"
                )
                if anomaly_cols:
                    with st.spinner("Detecting anomalies..."):
                        try:
                            anomalies = detect_anomalies(
                                df[available_columns], anomaly_cols, contamination)
                            st.write("Anomalies Detected:")
                            st.json(anomalies)
                        except Exception as e:
                            st.error(f"Error detecting anomalies: {str(e)}")

        with ml_container:
            with st.expander("One-Click ML Deployment", expanded=False):
                st.markdown(
                    '<span title="Train a machine learning model and deploy it">ℹ️</span>',
                    unsafe_allow_html=True)
                target_col = st.selectbox(
                    "Target Column (to predict)",
                    available_columns,
                    help="Column to predict with ML",
                    key="target_col"
                )
                feature_cols = st.multiselect(
                    "Feature Columns",
                    [col for col in available_columns if col != target_col],
                    help="Columns to use as predictors",
                    key="feature_cols"
                )
                train_ml = st.checkbox(
                    "Train and Deploy ML Model",
                    help="Generate a prediction app",
                    key="train_ml")
                if train_ml and not (target_col and feature_cols):
                    st.warning(
                        "Please select a target column and at least one feature column for ML deployment.")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            preview_button = st.form_submit_button(label="Preview Changes")
        with col2:
            apply_button = st.form_submit_button(label="Apply Changes")
        with col3:
            auto_clean_button = st.form_submit_button(label="Auto-Clean")

    # Handle form submission
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
            st.warning(
                "Please select at least one cleaning operation, custom rule, or ML deployment with valid parameters.")
        else:
            with st.spinner("Processing cleaning operations..."):
                try:
                    if replace_value.strip() and replace_with:
                        if replace_with == "Custom" and not replace_with.strip():
                            st.error(
                                "Please provide a custom replacement value.")
                            return
                        if replace_scope not in [
                                "All columns", "Numeric columns", "Categorical columns"]:
                            st.error("Invalid replacement scope selected.")
                            return

                    logger.info("Applying cleaning operations...")
                    cleaned_df, logs = apply_cleaning_operations(
                        df, selected_suggestions, columns_to_drop, options,
                        replace_value, replace_with if replace_with != "NaN" else "NaN",
                        replace_scope, encode_cols, encode_method, auto_clean=auto_clean_button,
                        enrich_col=enrich_col if enrich_col != "None" else None, enrich_api_key=enrich_api_key,
                        train_ml=train_ml, target_col=target_col, feature_cols=feature_cols
                    )
                    logger.info(f"Cleaning operations applied. Logs: {logs}")

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
                            else:
                                mask = cleaned_df[col] == threshold

                            if action == "Set to NaN":
                                cleaned_df.loc[mask, col] = pd.NA
                            else:
                                cleaned_df.loc[mask, col] = action_value

                            logs.append(
                                f"Applied custom rule on {col}: {condition} {threshold}, {action} {
                                    'NaN' if action == 'Set to NaN' else action_value}")

                    if preview_button:
                        st.subheader("Preview of Changes")
                        st.write("Before:")
                        st.dataframe(df.head(10), use_container_width=True)
                        st.write("After:")
                        st.dataframe(
                            cleaned_df.head(10),
                            use_container_width=True)
                        st.write("Preview Logs:")
                        for log in logs:
                            st.write(f"- {log}")

                    if apply_button or auto_clean_button:
                        st.session_state.previous_states.append(
                            (df.copy(), st.session_state.logs.copy()))
                        if len(st.session_state.previous_states) > 5:
                            st.session_state.previous_states.pop(0)
                        st.session_state.redo_states = []
                        st.session_state.cleaned_df = cleaned_df
                        st.session_state.logs = logs
                        if columns_to_drop:
                            st.session_state.dropped_columns.extend(
                                columns_to_drop)
                        st.session_state.cleaning_history.append({
                            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "logs": logs
                        })
                        st.session_state.suggestions = get_cached_suggestions(
                            cleaned_df[[col for col in cleaned_df.columns if col not in st.session_state.dropped_columns]])
                        st.success("Changes applied successfully!")
                        st.session_state.progress["Clean"] = "Done"
                        display_cleaned_dataset(st.session_state.cleaned_df)
                        from app import save_auth_state
                        save_auth_state()
                except Exception as e:
                    st.error(f"Error processing cleaning operations: {str(e)}")
                    logger.error(
                        f"Error in apply_cleaning_operations: {
                            str(e)}")
                    st.session_state.progress["Clean"] = "Failed"

    with st.expander("Save/Apply Cleaning Templates", expanded=False):
        st.subheader("Save/Apply Cleaning Templates")
        st.markdown(
            '<span title="Save your cleaning configuration as a template to reuse later">ℹ️</span>',
            unsafe_allow_html=True)
        with st.form(key="template_form"):
            template_name = st.text_input(
                "Template Name",
                "",
                help="Enter a name to save this cleaning configuration",
                key="template_name")
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
                st.session_state.cleaning_templates[template_name] = {
                    k: v for k, v in template.items() if k != "enrich_api_key"}
                st.success(f"Saved template '{template_name}'")

        if st.session_state.cleaning_templates:
            with st.form(key="apply_template_form"):
                template_to_apply = st.selectbox(
                    "Apply Saved Template",
                    ["None"] +
                    list(
                        st.session_state.cleaning_templates.keys()),
                    key="apply_template")
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
                            st.session_state.previous_states.append(
                                (df.copy(), st.session_state.logs.copy()))
                            if len(st.session_state.previous_states) > 5:
                                st.session_state.previous_states.pop(0)
                            st.session_state.redo_states = []
                            st.session_state.cleaned_df = cleaned_df
                            st.session_state.logs = logs
                            if template["columns_to_drop"]:
                                st.session_state.dropped_columns.extend(
                                    template["columns_to_drop"])
                            st.session_state.cleaning_history.append({
                                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                "logs": logs + [f"Applied template '{template_to_apply}'"]
                            })
                            st.session_state.suggestions = get_cached_suggestions(
                                cleaned_df[[col for col in cleaned_df.columns if col not in st.session_state.dropped_columns]])
                            st.success(
                                f"Applied template '{template_to_apply}'")
                            st.session_state.progress["Clean"] = "Done"
                            display_cleaned_dataset(
                                st.session_state.cleaned_df)
                            from app import save_auth_state
                            save_auth_state()
                        except Exception as e:
                            st.error(f"Error applying template: {str(e)}")
                            st.session_state.progress["Clean"] = "Failed"

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.previous_states and st.button(
                "Undo Last Cleaning", help="Revert to the previous state"):
            current_state = (
                st.session_state.cleaned_df.copy(),
                st.session_state.logs.copy())
            st.session_state.redo_states.append(current_state)
            if len(st.session_state.redo_states) > 5:
                st.session_state.redo_states.pop(0)
            previous_df, previous_logs = st.session_state.previous_states.pop()
            st.session_state.cleaned_df = previous_df
            st.session_state.logs = previous_logs
            st.session_state.suggestions = get_cached_suggestions(
                previous_df[[col for col in previous_df.columns if col not in st.session_state.dropped_columns]])
            st.session_state.cleaning_history.append({
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "logs": ["Undid last cleaning operation"]
            })
            st.rerun()

    with col2:
        if st.session_state.redo_states and st.button(
                "Redo Last Cleaning", help="Reapply the last undone state"):
            st.session_state.previous_states.append(
                (st.session_state.cleaned_df.copy(), st.session_state.logs.copy()))
            if len(st.session_state.previous_states) > 5:
                st.session_state.previous_states.pop(0)
            redo_df, redo_logs = st.session_state.redo_states.pop()
            st.session_state.cleaned_df = redo_df
            st.session_state.logs = redo_logs
            st.session_state.suggestions = get_cached_suggestions(
                redo_df[[col for col in redo_df.columns if col not in st.session_state.dropped_columns]])
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
        st.markdown(
            '<span title="Export your cleaned dataset as a CSV file for use in Tableau">ℹ️</span>',
            unsafe_allow_html=True)
        if st.session_state.cleaned_df is not None:
            export_button = st.button("Export Cleaned Dataset for Tableau")
            if export_button:
                filename = f"cleaned_for_tableau_{
                    datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                st.markdown(
                    get_download_link(
                        st.session_state.cleaned_df,
                        filename),
                    unsafe_allow_html=True)
                st.info(
                    "Download the CSV and import it into Tableau Public or Desktop to create visualizations!")


def render_insights_page() -> None:
    st.title("Insights Dashboard")
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("Please upload a dataset first on the Upload page.")
        return

    st.session_state.progress["Insights"] = "In Progress"

    df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df
    available_columns = [
        col for col in df.columns if col not in st.session_state.dropped_columns]

    with st.spinner("Generating insights..."):
        try:
            insights = get_insights(df[available_columns])
            st.subheader("Key Insights")
            for insight in insights:
                st.write(f"- {insight}")
            st.session_state.progress["Insights"] = "Done"
        except Exception as e:
            st.error(f"Error generating insights: {str(e)}")
            st.session_state.progress["Insights"] = "Failed"


def render_predictive_page(df: pd.DataFrame) -> None:
    st.title("Predictive Analytics")
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("Please upload a dataset first on the Upload page.")
        return

    st.session_state.progress["Predictive"] = "In Progress"

    available_columns = [
        col for col in df.columns if col not in st.session_state.dropped_columns]
    df = df[available_columns]

    st.subheader("Predictive Dashboard")
    render_predictive_page_external(df)

    st.subheader("Generate Synthetic Data")
    task_type = st.selectbox(
        "Task Type", [
            "classification", "regression"], key="task_type")
    if st.button("Generate Synthetic Data", key="generate_synthetic"):
        with st.spinner("Generating synthetic data..."):
            try:
                synthetic_df = generate_synthetic_data(df, task_type)
                st.session_state.cleaned_df = synthetic_df
                st.session_state.suggestions = get_cached_suggestions(
                    synthetic_df[[col for col in synthetic_df.columns if col not in st.session_state.dropped_columns]])
                st.session_state.cleaning_history.append({
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "logs": ["Generated synthetic data"]
                })
                st.write("Synthetic Dataset Preview:")
                st.dataframe(synthetic_df.head(10), use_container_width=True)
                st.markdown(get_download_link(synthetic_df,
                                              f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
                            unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error generating synthetic data: {str(e)}")
                st.session_state.progress["Predictive"] = "Failed"

    st.subheader("Time Series Forecasting")
    time_cols = [
        col for col in df.columns if pd.api.types.is_datetime64_any_dtype(
            df[col])]
    if time_cols:
        forecast_col = st.selectbox(
            "Select time series column",
            time_cols,
            key="forecast_col")
        periods = st.slider("Forecast periods", 1, 30,
                            5, key="forecast_periods")
        freq = st.selectbox("Frequency",
                            ["D",
                             "M",
                             "Y"],
                            help="Select the frequency of the time series data",
                            key="forecast_freq")
        if st.button("Forecast", key="forecast_button"):
            with st.spinner("Forecasting..."):
                try:
                    forecast_df = forecast_time_series(
                        df, forecast_col, periods, time_col=forecast_col, freq=freq)
                    st.write("Forecasted Values:")
                    st.dataframe(forecast_df, use_container_width=True)
                    st.markdown(get_download_link(forecast_df,
                                                  f"forecast_{forecast_col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
                                unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error forecasting time series: {str(e)}")
                    st.session_state.progress["Predictive"] = "Failed"
    else:
        st.info("No datetime columns found for time series forecasting.")

    st.subheader("Time Series Decomposition")
    if time_cols:
        decompose_col = st.selectbox(
            "Select column for decomposition",
            time_cols,
            key="decompose_col")
        period = st.slider("Period for decomposition", 1,
                           30, 12, key="decompose_period")
        if st.button("Decompose Time Series", key="decompose_button"):
            with st.spinner("Decomposing time series..."):
                try:
                    decomposition = analyze_time_series(
                        df, decompose_col, period)
                    if decomposition:
                        st.write("Trend Component:")
                        st.line_chart(decomposition.get("trend"))
                        st.write("Seasonal Component:")
                        st.line_chart(decomposition.get("seasonal"))
                        st.write("Residual Component:")
                        st.line_chart(decomposition.get("residual"))
                    else:
                        st.error("Failed to decompose time series")
                        st.session_state.progress["Predictive"] = "Failed"
                except Exception as e:
                    st.error(f"Error decomposing time series: {str(e)}")
                    st.session_state.progress["Predictive"] = "Failed"
    else:
        st.info("No datetime columns found for time series decomposition.")

    st.subheader("Clustering")
    numeric_cols = df.select_dtypes(
        include=['int64', 'float64']).columns.tolist()
    cluster_cols = st.multiselect(
        "Select columns for clustering",
        numeric_cols,
        key="cluster_cols")
    n_clusters = st.slider("Number of clusters", 2, 10, 3, key="n_clusters")
    if st.button("Perform Clustering", key="ui_perform_clustering"):
        if len(cluster_cols) < 2:
            st.warning("Please select at least two columns for clustering")
        else:
            with st.spinner("Performing clustering..."):
                try:
                    labels = perform_clustering(df, cluster_cols, n_clusters)
                    df['Cluster'] = labels
                    st.session_state.cleaned_df = df
                    st.session_state.suggestions = get_cached_suggestions(
                        df[[col for col in df.columns if col not in st.session_state.dropped_columns]])
                    st.session_state.cleaning_history.append({
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "logs": ["Performed clustering"]
                    })
                    st.write("Dataset with Cluster Labels:")
                    st.dataframe(df.head(10), use_container_width=True)
                    st.markdown(get_download_link(df,
                                                  f"clustered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
                                unsafe_allow_html=True)
                    st.session_state.progress["Predictive"] = "Done"
                except Exception as e:
                    st.error(f"Error performing clustering: {str(e)}")
                    st.session_state.progress["Predictive"] = "Failed"
