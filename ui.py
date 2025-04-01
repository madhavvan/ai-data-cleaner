import streamlit as st
import pandas as pd
from datetime import datetime
import base64
import json
import os
from data_utils import (
    get_cleaning_suggestions, apply_cleaning_operations, extract_column, 
    calculate_health_score, chat_with_gpt, detect_anomalies, get_insights, 
    suggest_workflow, train_ml_model, forecast_time_series, perform_clustering, 
    generate_synthetic_data, analyze_time_series
)
from predictive import render_predictive_page as render_predictive_page_external

# Cache expensive operations
@st.cache_data
def get_cached_suggestions(df):
    return get_cleaning_suggestions(df)

def get_download_link(df, filename):
    """Generate a download link for the cleaned dataset."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download cleaned dataset</a>'

def render_upload_page():
    """Render the upload page UI with session persistence and chunked processing."""
    st.title("📤 Upload Your Dataset")
    st.markdown("<p class='welcome'>Start your data journey here!</p>", unsafe_allow_html=True)

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
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'cleaning_history' not in st.session_state:
        st.session_state.cleaning_history = []
    if 'cleaning_templates' not in st.session_state:
        st.session_state.cleaning_templates = {}
    if 'is_premium' not in st.session_state:
        st.session_state.is_premium = False
    if 'ai_suggestions_used' not in st.session_state:
        st.session_state.ai_suggestions_used = 0

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

    uploaded_file = st.file_uploader("Choose a file (CSV or Excel)", type=["csv", "xlsx"], help="Upload a CSV or Excel file to begin.")
    if uploaded_file:
        try:
            if uploaded_file.size > 50 * 1024 * 1024:  # 50MB
                st.warning("File size exceeds 50MB. Using chunked processing.")
                if uploaded_file.name.endswith('.csv'):
                    chunks = pd.read_csv(uploaded_file, chunksize=10000)
                    df_list = []
                    progress_bar = st.progress(0)  # Enhancement: Add progress bar
                    total_chunks = uploaded_file.size // (10000 * 100)  # Rough estimate
                    for i, chunk in enumerate(chunks):
                        df_list.append(chunk)
                        progress_bar.progress(min((i + 1) / total_chunks, 1.0))
                    df = pd.concat(df_list, ignore_index=True)
                else:
                    df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

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

            st.subheader("Dataset Preview (First 10 Rows)")
            st.dataframe(df.head(10))
            st.subheader("Basic Metadata")
            score = calculate_health_score(df)
            st.write(f"Rows: {df.shape[0]}")
            st.write(f"Columns: {df.shape[1]}")
            st.write(f"Missing Values: {df.isna().sum().sum()}")
            st.progress(score / 100)
            st.write(f"Dataset Health Score: {score}/100")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

def render_clean_page():
    """Render the clean page UI with robust multi-change logic and enhanced UX."""
    st.title("🧹 Clean Your Dataset")
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("Please upload a dataset first on the Upload page.")
        return

    # Always use the latest cleaned dataset as the base
    df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df

    # Initialize suggestions if not present (run silently in the background)
    if not st.session_state.suggestions or id(st.session_state.cleaned_df) != id(df):
        st.session_state.suggestions = get_cached_suggestions(df)

    st.subheader("Dataset Health")
    score = calculate_health_score(df)
    st.progress(score / 100)
    st.write(f"Current Health Score: {score}/100")

    # Smart Workflow Automation
    st.subheader("Smart Workflow Automation")
    if st.button("Run Smart Workflow"):
        with st.spinner("Generating and executing workflow..."):
            try:
                workflow = suggest_workflow(df)
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
                # Update suggestions silently
                st.session_state.suggestions = get_cached_suggestions(cleaned_df)
                st.success("Smart Workflow executed successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error executing smart workflow: {str(e)}")

    # Cleaning Form with Robust Logic
    with st.form(key="cleaning_form", clear_on_submit=False):
        manual_container = st.container()
        replace_container = st.container()
        encode_container = st.container()
        enrich_container = st.container()
        ai_container = st.container()
        anomaly_container = st.container()
        ml_container = st.container()

        # Initialize all variables to avoid UnboundLocalError
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

        with manual_container:
            st.subheader("Manual Column Dropping")
            columns_to_drop = st.multiselect("Select columns to drop", df.columns.tolist(), 
                                           help="Choose columns to remove from the dataset.")
        
        with replace_container:
            with st.expander("Custom Value Replacement", expanded=True):
                st.markdown("**Replace unwanted values (e.g., '?' for missing data)**")
                replace_value = st.text_input("Value to replace (e.g., ?, 999, Unknown)", "", 
                                            help="Enter the value you want to replace. For example, use '?' to replace missing value markers.")
                replace_with = st.radio("Replace with", ["NaN", "?", "0", "Custom"], 
                                      help="Select what to replace the value with. 'NaN' is recommended for missing data.")
                if replace_with == "Custom":
                    replace_with = st.text_input("Custom replacement value", "", 
                                               help="Enter a custom replacement value.")
                replace_scope = st.radio("Apply to", ["All columns", "Numeric columns", "Categorical columns"], 
                                       help="Choose which columns to apply the replacement to.")
        
        with encode_container:
            with st.expander("Convert Categorical to Numerical", expanded=False):
                cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                encode_cols = st.multiselect("Select categorical columns to convert", cat_cols, 
                                           help="Choose categorical columns to convert to numerical.")
                encode_method = st.radio("Conversion method", ["Label Encoding", "One-Hot Encoding"], 
                                       help="Label Encoding assigns integers; One-Hot creates dummy columns.")
        
        with enrich_container:
            with st.expander("Smart Data Enrichment", expanded=False):
                enrich_col = st.selectbox("Column to Enrich (e.g., address)", ["None"] + df.columns.tolist(), 
                                        help="Select a column to enrich with external data.")
                enrich_api_key = st.text_input("Google API Key (for geolocation)", type="password", 
                                             help="Enter your Google Maps API key.")
                if enrich_col != "None" and not enrich_api_key:
                    st.warning("Google API Key is required for data enrichment.")
        
        with ai_container:
            with st.expander("AI Cleaning Suggestions", expanded=True):
                for suggestion, explanation in st.session_state.suggestions:
                    if st.checkbox(f"{suggestion} - {explanation}", key=suggestion):
                        selected_suggestions.append((suggestion, explanation))
                        st.session_state.ai_suggestions_used += 1
                        if "Handle special characters" in suggestion:
                            options["special_chars"] = st.radio("Action for special characters", 
                                                              ("Drop them", "Replace with underscores"), 
                                                              key=f"special_chars_opt_{suggestion}")
                        elif "Fill missing values" in suggestion:
                            col = extract_column(suggestion)
                            if col and col in df.columns and df[col].dtype in ['int64', 'float64']:
                                options[f"fill_{col}"] = st.radio(f"Fill method for {col}", 
                                                                ["mean", "median", "mode"], 
                                                                key=f"fill_opt_{col}_{suggestion}")
                        elif "Handle outliers" in suggestion:
                            col = extract_column(suggestion)
                            if col and col in df.columns:
                                options[f"outlier_{col}"] = st.radio(f"Action for outliers in {col}", 
                                                                   ("Remove", "Cap at bounds"), 
                                                                   key=f"outlier_opt_{col}_{suggestion}")
        
        with anomaly_container:
            with st.expander("Anomaly Detection", expanded=False):
                num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                anomaly_cols = st.multiselect("Select numerical columns for anomaly detection", num_cols, 
                                            help="Detect outliers using AI.")
                if anomaly_cols:
                    with st.spinner("Detecting anomalies..."):
                        try:
                            anomalies = detect_anomalies(df, anomaly_cols)
                            st.write("Anomalies Detected:")
                            st.json(anomalies)
                        except Exception as e:
                            st.error(f"Error detecting anomalies: {str(e)}")

        with ml_container:
            with st.expander("One-Click ML Deployment", expanded=False):
                target_col = st.selectbox("Target Column (to predict)", df.columns.tolist(), 
                                        help="Column to predict with ML.")
                feature_cols = st.multiselect("Feature Columns", df.columns.tolist(), 
                                            help="Columns to use as predictors.")
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

    # Process cleaning operations with robust validation
    if preview_button or apply_button or auto_clean_button:
        operations_selected = (
            selected_suggestions or
            columns_to_drop or
            (replace_value.strip() and replace_with) or
            encode_cols or
            (enrich_col != "None" and enrich_api_key) or
            auto_clean_button or
            (train_ml and target_col and feature_cols)
        )
        if not operations_selected:
            st.warning("Please select at least one cleaning operation or ML deployment with valid parameters.")
        else:
            with st.spinner("Processing..."):
                try:
                    cleaned_df, logs = apply_cleaning_operations(
                        df, selected_suggestions, columns_to_drop, options, 
                        replace_value, replace_with if replace_with != "NaN" else "NaN", 
                        replace_scope, encode_cols, encode_method, auto_clean=auto_clean_button, 
                        enrich_col=enrich_col if enrich_col != "None" else None, enrich_api_key=enrich_api_key,
                        train_ml=train_ml, target_col=target_col, feature_cols=feature_cols
                    )
                    
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
                        st.session_state.cleaning_history.append({
                            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "logs": logs
                        })
                        # Update suggestions silently after applying changes
                        st.session_state.suggestions = get_cached_suggestions(cleaned_df)
                        st.success("Changes applied successfully!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error processing cleaning operations: {str(e)}")

    # Template Management
    with st.expander("Save/Apply Cleaning Templates", expanded=False):
        st.subheader("Save/Apply Cleaning Templates")
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
                    "feature_cols": feature_cols
                }
                st.session_state.cleaning_templates[template_name] = template
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
                            st.session_state.cleaning_history.append({
                                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                "logs": logs + [f"Applied template '{template_to_apply}'"]
                            })
                            # Update suggestions silently
                            st.session_state.suggestions = get_cached_suggestions(cleaned_df)
                            st.success(f"Applied template '{template_to_apply}'")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error applying template: {str(e)}")

    # Undo/Redo Buttons
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
            # Update suggestions silently
            st.session_state.suggestions = get_cached_suggestions(previous_df)
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
            # Update suggestions silently
            st.session_state.suggestions = get_cached_suggestions(redo_df)
            st.session_state.cleaning_history.append({
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "logs": ["Redid last cleaning operation"]
            })
            st.rerun()

    # Display Cleaning History
    with st.expander("Cleaning History", expanded=False):
        st.subheader("Cleaning History")
        if st.session_state.cleaning_history:
            for entry in st.session_state.cleaning_history:
                st.write(f"**{entry['timestamp']}**")
                for log in entry['logs']:
                    st.write(f"- {log}")
        else:
            st.write("No cleaning operations have been performed yet.")

    # Export to Tableau
    with st.expander("Export to Tableau", expanded=False):
        st.subheader("Export to Tableau")
        if st.session_state.cleaned_df is not None:
            export_button = st.button("Export Cleaned Dataset for Tableau")
            if export_button:
                filename = f"cleaned_for_tableau_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                st.markdown(get_download_link(st.session_state.cleaned_df, filename), unsafe_allow_html=True)
                st.info("Download the CSV and import it into Tableau Public or Desktop to create visualizations!")

    # Enhancement: Add pagination for large datasets
    if st.session_state.cleaned_df is not None:
        st.subheader("Cleaned Dataset Preview")
        view_option = st.radio("View dataset as:", ("First 10 Rows", "Full Dataset"), horizontal=True)
        if view_option == "First 10 Rows":
            st.dataframe(st.session_state.cleaned_df.head(10))
        else:
            page_size = 100
            total_rows = len(st.session_state.cleaned_df)
            total_pages = (total_rows + page_size - 1) // page_size
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, total_rows)
            st.dataframe(st.session_state.cleaned_df.iloc[start_idx:end_idx], use_container_width=True)
        
        st.subheader("Cleaning Summary")
        st.write(f"Original Shape: {st.session_state.df.shape}")
        st.write(f"New Shape: {st.session_state.cleaned_df.shape}")
        st.write(f"New Health Score: {calculate_health_score(st.session_state.cleaned_df)}/100")
        for log in st.session_state.logs:
            st.write(f"- {log}")
        st.markdown(get_download_link(st.session_state.cleaned_df, 
                                    f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), 
                   unsafe_allow_html=True)

def render_insights_page():
    """Render the insights page with NLG."""
    st.title("💡 Insights Dashboard")
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("Please upload a dataset first on the Upload page.")
        return

    df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df

    with st.spinner("Generating insights..."):
        try:
            insights = get_insights(df)
            st.subheader("Key Insights")
            for insight in insights:
                st.write(f"- {insight}")
        except Exception as e:
            st.error(f"Error generating insights: {str(e)}")

def render_predictive_page(df):
    """Render the predictive analytics page with ML model training, forecasting, and clustering."""
    st.title("🔮 Predictive Analytics")
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("Please upload a dataset first on the Upload page.")
        return

    st.subheader("Predictive Dashboard")
    render_predictive_page_external(df)

    st.subheader("Generate Synthetic Data")
    task_type = st.selectbox("Task Type", ["classification", "regression"])
    if st.button("Generate Synthetic Data"):
        with st.spinner("Generating synthetic data..."):
            try:
                synthetic_df = generate_synthetic_data(df, task_type)
                st.session_state.cleaned_df = synthetic_df
                # Update suggestions silently
                st.session_state.suggestions = get_cached_suggestions(synthetic_df)
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
                st.error(f"Error generating synthetic data: {str(e)}")

    st.subheader("Time Series Forecasting")
    time_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    if time_cols:
        forecast_col = st.selectbox("Select time series column", time_cols)
        periods = st.slider("Forecast periods", 1, 30, 5)
        if st.button("Forecast"):
            with st.spinner("Forecasting..."):
                try:
                    forecast_df = forecast_time_series(df, forecast_col, periods, time_col=forecast_col)  # Fix: Pass time_col
                    st.write("Forecasted Values:")
                    st.dataframe(forecast_df)
                    st.markdown(get_download_link(forecast_df, 
                                                f"forecast_{forecast_col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), 
                               unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error forecasting time series: {str(e)}")
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
                except Exception as e:
                    st.error(f"Error decomposing time series: {str(e)}")
    else:
        st.info("No datetime columns found for time series decomposition.")

    st.subheader("Clustering")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cluster_cols = st.multiselect("Select columns for clustering", numeric_cols)
    n_clusters = st.slider("Number of clusters", 2, 10, 3)
    if st.button("Perform Clustering", key="ui_perform_clustering"):  # Added unique key
        if len(cluster_cols) < 2:
            st.warning("Please select at least two columns for clustering.")
        else:
            with st.spinner("Performing clustering..."):
                try:
                    labels = perform_clustering(df, cluster_cols, n_clusters)
                    df['Cluster'] = labels
                    st.session_state.cleaned_df = df
                    # Update suggestions silently
                    st.session_state.suggestions = get_cached_suggestions(df)
                    st.session_state.cleaning_history.append({
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "logs": ["Performed clustering"]
                    })
                    st.write("Dataset with Cluster Labels:")
                    st.dataframe(df.head(10))
                    st.markdown(get_download_link(df, 
                                                f"clustered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), 
                               unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error performing clustering: {str(e)}")