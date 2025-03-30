import streamlit as st
import pandas as pd
from datetime import datetime
import base64
import json
from data_utils import get_cleaning_suggestions, apply_cleaning_operations, extract_column, calculate_health_score, chat_with_gpt

def get_download_link(df, filename):
    """Generate a download link for the cleaned dataset."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download cleaned dataset</a>'

def render_upload_page():
    """Render the upload page UI with session persistence."""
    st.title("📤 Upload Your Dataset")
    st.markdown("<p class='welcome'>Start your data journey here!</p>", unsafe_allow_html=True)

    # Initialize session state variables if not present
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

    # Display current dataset if it exists
    if st.session_state.df is not None:
        st.subheader("Current Dataset Preview (First 10 Rows)")
        st.dataframe(st.session_state.df.head(10))
        st.subheader("Basic Metadata")
        score = calculate_health_score(st.session_state.df)
        st.write(f"Rows: {st.session_state.df.shape[0]}")
        st.write(f"Columns: {st.session_state.df.shape[1]}")
        st.write(f"Missing Values: {st.session_state.df.isna().sum().sum()}")
        st.progress(score / 100)
        st.write(f"Dataset Health Score: {score}/100")

        # Warn user about overwriting
        st.warning("Uploading a new file will overwrite the current dataset and reset all cleaning operations. Proceed with caution!")

    # File uploader
    uploaded_file = st.file_uploader("Choose a file (CSV or Excel)", type=["csv", "xlsx"], help="Upload a CSV or Excel file to begin.")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            # Reset session state when a new file is uploaded
            st.session_state.df = df
            st.session_state.cleaned_df = None
            st.session_state.logs = []
            st.session_state.suggestions = []
            st.session_state.previous_states = []
            st.session_state.redo_states = []
            st.session_state.chat_history = []
            st.session_state.cleaning_history = []
            st.session_state.cleaning_templates = {}

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
    """Render the clean page UI with improved re-editing logic and bonus features."""
    st.title("🧹 Clean Your Dataset")
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("Please upload a dataset first on the Upload page.")
        return

    # Use the cleaned dataset if it exists, otherwise use the original
    df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df

    # Get AI cleaning suggestions if not already fetched
    if not st.session_state.suggestions:
        with st.spinner("Analyzing dataset with GPT-4o..."):
            st.session_state.suggestions = get_cleaning_suggestions(df)
    
    st.subheader("Dataset Health")
    score = calculate_health_score(df)
    st.progress(score / 100)
    st.write(f"Current Health Score: {score}/100")

    # Chatbot Interface
    st.subheader("AI Data Assistant")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.get('chat_history', []):
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    chat_input = st.chat_input("Ask me anything about your data!")
    if chat_input:
        st.session_state.chat_history.append({"role": "user", "content": chat_input})
        response = chat_with_gpt(df, chat_input)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    # Cleaning Form
    with st.form(key="cleaning_form"):
        manual_container = st.container()
        replace_container = st.container()
        encode_container = st.container()
        enrich_container = st.container()
        ai_container = st.container()
        ml_container = st.container()
        template_container = st.container()

        with manual_container:
            st.subheader("Manual Column Dropping")
            columns_to_drop = st.multiselect("Select columns to drop", df.columns.tolist(), 
                                           help="Choose columns to remove from the dataset.")
        
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
        
        with ai_container:
            with st.expander("AI Cleaning Suggestions", expanded=True):
                selected_suggestions = []
                options = {}
                for suggestion, explanation in st.session_state.suggestions:
                    if st.checkbox(f"{suggestion} - {explanation}", key=suggestion):
                        selected_suggestions.append((suggestion, explanation))
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
        
        with ml_container:
            with st.expander("One-Click ML Deployment", expanded=False):
                target_col = st.selectbox("Target Column (to predict)", df.columns.tolist(), 
                                        help="Column to predict with ML.")
                feature_cols = st.multiselect("Feature Columns", df.columns.tolist(), 
                                            help="Columns to use as predictors.")
                train_ml = st.checkbox("Train and Deploy ML Model", help="Generate a prediction app.")

        with template_container:
            with st.expander("Save/Apply Cleaning Templates", expanded=False):
                template_name = st.text_input("Template Name", "", help="Enter a name to save this cleaning configuration as a template.")
                if st.button("Save as Template") and template_name:
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
                    template_to_apply = st.selectbox("Apply Saved Template", ["None"] + list(st.session_state.cleaning_templates.keys()))
                    if template_to_apply != "None":
                        template = st.session_state.cleaning_templates[template_to_apply]
                        columns_to_drop = template["columns_to_drop"]
                        selected_suggestions = template["selected_suggestions"]
                        options = template["options"]
                        replace_value = template["replace_value"]
                        replace_with = template["replace_with"]
                        replace_scope = template["replace_scope"]
                        encode_cols = template["encode_cols"]
                        encode_method = template["encode_method"]
                        enrich_col = template["enrich_col"]
                        train_ml = template["train_ml"]
                        target_col = template["target_col"]
                        feature_cols = template["feature_cols"]
                        st.info(f"Applied template '{template_to_apply}'")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            preview_button = st.form_submit_button(label="Preview Changes")
        with col2:
            apply_button = st.form_submit_button(label="Apply Changes")
        with col3:
            auto_clean_button = st.form_submit_button(label="Auto-Clean")

    # Process cleaning operations
    if (preview_button or apply_button or auto_clean_button) and (selected_suggestions or columns_to_drop or replace_value or encode_cols or enrich_col != "None" or auto_clean_button or train_ml):
        with st.spinner("Processing..."):
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
                # Save the current state for undo
                if st.session_state.cleaned_df is not None:
                    st.session_state.previous_states.append((st.session_state.cleaned_df.copy(), st.session_state.logs.copy()))
                else:
                    st.session_state.previous_states.append((st.session_state.df.copy(), []))
                if len(st.session_state.previous_states) > 5:
                    st.session_state.previous_states.pop(0)
                
                # Clear redo stack on new action
                st.session_state.redo_states = []

                # Update cleaned_df and logs
                st.session_state.cleaned_df = cleaned_df
                st.session_state.logs = logs

                # Update cleaning history
                st.session_state.cleaning_history.append({
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "logs": logs
                })

                # Refresh AI suggestions for the updated dataset
                with st.spinner("Refreshing AI suggestions..."):
                    st.session_state.suggestions = get_cleaning_suggestions(cleaned_df)

    # Undo/Redo Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.get('previous_states') and st.button("Undo Last Cleaning", help="Revert to the previous state"):
            # Save current state for redo
            current_state = (st.session_state.cleaned_df.copy(), st.session_state.logs.copy())
            st.session_state.redo_states.append(current_state)
            if len(st.session_state.redo_states) > 5:
                st.session_state.redo_states.pop(0)
            
            # Restore previous state
            previous_df, previous_logs = st.session_state.previous_states.pop()
            st.session_state.cleaned_df = previous_df
            st.session_state.logs = previous_logs
            st.session_state.suggestions = get_cleaning_suggestions(previous_df)
            st.session_state.cleaning_history.append({
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "logs": ["Undid last cleaning operation"]
            })
            st.rerun()

    with col2:
        if st.session_state.get('redo_states') and st.button("Redo Last Cleaning", help="Reapply the last undone state"):
            # Save current state for undo
            st.session_state.previous_states.append((st.session_state.cleaned_df.copy(), st.session_state.logs.copy()))
            if len(st.session_state.previous_states) > 5:
                st.session_state.previous_states.pop(0)
            
            # Restore redo state
            redo_df, redo_logs = st.session_state.redo_states.pop()
            st.session_state.cleaned_df = redo_df
            st.session_state.logs = redo_logs
            st.session_state.suggestions = get_cleaning_suggestions(redo_df)
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
                for log in entry["logs"]:
                    st.write(f"- {log}")
        else:
            st.write("No cleaning operations have been performed yet.")

    # Display Cleaned Dataset
    if st.session_state.get('cleaned_df') is not None:
        st.subheader("Cleaned Dataset Preview")
        st.dataframe(st.session_state.cleaned_df.head(10))
        
        st.subheader("Cleaning Summary")
        st.write(f"Original Shape: {st.session_state.df.shape}")
        st.write(f"New Shape: {st.session_state.cleaned_df.shape}")
        st.write(f"New Health Score: {calculate_health_score(st.session_state.cleaned_df)}/100")
        for log in st.session_state.logs:
            st.write(f"- {log}")
        
        st.markdown(get_download_link(st.session_state.cleaned_df, 
                                    f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), 
                   unsafe_allow_html=True)