import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from dotenv import load_dotenv
from openai import OpenAI
import io

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY') or st.secrets.get("OPENAI_API_KEY")

if openai_api_key is None:
    st.error("❌ OpenAI API Key not found.")
    st.stop()

client = OpenAI(api_key=openai_api_key)

st.set_page_config(page_title="Advanced AI Data Cleaner", layout="wide")
st.title('🚀 Advanced AI Data Cleaner & Analyzer')

# Session state setup
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = []
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None

# File uploader
uploaded_file = st.file_uploader('Upload CSV or Excel:', ['csv', 'xlsx'])

if uploaded_file:
    if st.session_state.df_original is None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

        # Replace '?' with NaN
        df.replace('?', np.nan, inplace=True)
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        st.session_state.df_original = df.copy()
        st.session_state.df_cleaned = df.copy()
    else:
        df = st.session_state.df_original

    st.success('✅ File uploaded successfully!')
    st.subheader('📌 Original Data Preview:')
    st.dataframe(df, use_container_width=True)

    if st.button('🔎 Generate AI Cleaning Suggestions'):
        with st.spinner('Generating AI suggestions...'):
            prompt_cleaning = f"""Analyze this dataset and provide specific data cleaning recommendations:
            {df.head().to_string()}
            
            Provide clear, actionable cleaning steps in bullet points format.
            For each recommendation, specify:
            - Exactly what should be done
            - Which columns it applies to (if specific)
            - What method to use (for handling missing values, etc.)
            """

            completion_cleaning = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt_cleaning}]
            )

            suggestions = completion_cleaning.choices[0].message.content
            st.session_state.suggestions = [s.strip('- ') for s in suggestions.strip().split('\n') if s]

    if st.session_state.suggestions:
        st.subheader('🧹 AI Cleaning Suggestions:')
        selected_steps = []
        for i, step in enumerate(st.session_state.suggestions):
            if st.checkbox(f"{i+1}. {step}", value=True, key=f"suggestion_{i}"):
                selected_steps.append(step)

# Cleaning logic
if st.button('✅ Apply Cleaning Steps') and 'df_cleaned' in st.session_state:
    cleaned_df = st.session_state.df_original.copy()
    actions_applied = []

    for step in selected_steps:
        step_lower = step.lower()

        # 0. Replace '?' with NaN
        if "replace '?'" in step_lower or "convert '?' to nan" in step_lower:
            before_nan = cleaned_df.isna().sum().sum()
            cleaned_df.replace('?', np.nan, inplace=True)
            after_nan = cleaned_df.isna().sum().sum()
            added_nans = after_nan - before_nan
            actions_applied.append(f"✅ Replaced '?' with NaN (added {added_nans} new missing values)")

        # 1. Remove rows with missing values
        elif any(phrase in step_lower for phrase in ['remove rows with missing', 'delete missing', 'drop na']):
            before = len(cleaned_df)
            cleaned_df.dropna(inplace=True)
            removed = before - len(cleaned_df)
            actions_applied.append(f"✅ Removed {removed} rows with missing values" if removed > 0 else "ℹ️ No rows with missing values found")

        # 2. Fill missing numerical values
        elif any(phrase in step_lower for phrase in ['fill missing numerical', 'fill missing values']):
            numeric_cols = cleaned_df.select_dtypes(include=['float64', 'int64']).columns
            filled_cols = []
            for col in numeric_cols:
                if cleaned_df[col].isna().any():
                    val = (
                        cleaned_df[col].mean() if 'mean' in step_lower else
                        cleaned_df[col].mode()[0] if 'mode' in step_lower else
                        cleaned_df[col].median()
                    )
                    cleaned_df[col].fillna(val, inplace=True)
                    filled_cols.append(col)
            actions_applied.append(f"✅ Filled missing values in: {', '.join(filled_cols)}" if filled_cols else "ℹ️ No numerical columns with missing values found")

        # 3. Fill missing categorical values
        elif any(phrase in step_lower for phrase in ['fill missing categorical', 'fill missing text']):
            cat_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns
            filled_cols = []
            for col in cat_cols:
                if cleaned_df[col].isna().any():
                    cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
                    filled_cols.append(col)
            actions_applied.append(f"✅ Filled missing values in: {', '.join(filled_cols)}" if filled_cols else "ℹ️ No categorical columns with missing values found")

        # 4. Remove duplicates
        elif 'remove duplicate' in step_lower:
            before = len(cleaned_df)
            cleaned_df.drop_duplicates(inplace=True)
            removed = before - len(cleaned_df)
            actions_applied.append(f"✅ Removed {removed} duplicate rows" if removed > 0 else "ℹ️ No duplicate rows found")

        # 5. Drop specific columns
        elif 'drop column' in step_lower or 'remove column' in step_lower:
            cols_to_drop = [col for col in cleaned_df.columns if col.lower() in step_lower or f"'{col}'" in step or f'"{col}"' in step]
            if cols_to_drop:
                cleaned_df.drop(columns=cols_to_drop, inplace=True)
                actions_applied.append(f"✅ Dropped columns: {', '.join(cols_to_drop)}")
            else:
                actions_applied.append("⚠️ Could not identify specific columns to drop")

        # 6. Convert data types
        elif any(phrase in step_lower for phrase in ['convert to numeric', 'change type']):
            actions_applied.append("ℹ️ Data type conversion requires explicit column info")

        # 7. Standardize or normalize
        elif any(phrase in step_lower for phrase in ['standardize', 'normalize']):
            numeric_cols = cleaned_df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                if 'standardize' in step_lower:
                    cleaned_df[numeric_cols] = (cleaned_df[numeric_cols] - cleaned_df[numeric_cols].mean()) / cleaned_df[numeric_cols].std()
                    actions_applied.append(f"✅ Standardized columns: {', '.join(numeric_cols)}")
                else:
                    cleaned_df[numeric_cols] = (cleaned_df[numeric_cols] - cleaned_df[numeric_cols].min()) / (cleaned_df[numeric_cols].max() - cleaned_df[numeric_cols].min())
                    actions_applied.append(f"✅ Normalized columns: {', '.join(numeric_cols)}")
            else:
                actions_applied.append("ℹ️ No numerical columns found to standardize/normalize")

        else:
            actions_applied.append(f"⚠️ Could not process suggestion: {step}")

    # Store cleaned version
    st.session_state.df_cleaned = cleaned_df
    st.success("✅ Cleaning steps applied successfully!")

    # Summary of actions
    st.subheader("📝 Applied Actions:")
    for action in actions_applied:
        if action.startswith("✅"):
            st.success(action)
        elif action.startswith("⚠️"):
            st.warning(action)
        else:
            st.info(action)

    # Show cleaned preview
    st.subheader("✨ Cleaned Data Preview:")
    st.dataframe(st.session_state.df_cleaned, use_container_width=True)

    # Compare columns and rows
    st.subheader("🔄 Changes Summary:")
    if len(st.session_state.df_original) != len(st.session_state.df_cleaned):
        st.info(f"Row count changed: {len(st.session_state.df_original)} → {len(st.session_state.df_cleaned)}")

    original_cols = set(st.session_state.df_original.columns)
    cleaned_cols = set(st.session_state.df_cleaned.columns)
    added = cleaned_cols - original_cols
    removed = original_cols - cleaned_cols
    if added:
        st.info(f"Added columns: {', '.join(added)}")
    if removed:
        st.info(f"Removed columns: {', '.join(removed)}")

    # Download cleaned data
    buffer = io.BytesIO()
    st.session_state.df_cleaned.to_csv(buffer, index=False)
    buffer.seek(0)
    st.download_button("⬇️ Download Cleaned Data", data=buffer, file_name="cleaned_data.csv", mime="text/csv")

    # AI Insight generation
    with st.spinner('Generating insights...'):
        prompt_insights = f"""Analyze this cleaned dataset and provide key insights:
        {st.session_state.df_cleaned.head().to_string()}
        
        Provide:
        1. Key statistics and patterns
        2. Notable observations
        3. Potential relationships between variables
        4. Any data quality issues remaining
        """

        completion_insights = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt_insights}]
        )

        insights = completion_insights.choices[0].message.content
        st.subheader("📈 AI Insights:")
        st.write(insights)

        numeric_columns = st.session_state.df_cleaned.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_columns) >= 2:
            st.subheader("📊 Auto-Generated Visualizations")
            col1, col2 = st.columns(2)

            with col1:
                try:
                    fig = px.scatter(st.session_state.df_cleaned, x=numeric_columns[0], y=numeric_columns[1],
                                     title=f"{numeric_columns[0]} vs {numeric_columns[1]}")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not create scatter plot: {str(e)}")

            with col2:
                try:
                    fig2 = px.histogram(st.session_state.df_cleaned, x=numeric_columns[0],
                                        title=f"Distribution of {numeric_columns[0]}")
                    st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not create histogram: {str(e)}")
