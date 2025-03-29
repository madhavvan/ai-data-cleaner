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

if 'suggestions' not in st.session_state:
    st.session_state.suggestions = []

uploaded_file = st.file_uploader('Upload CSV or Excel:', ['csv', 'xlsx'])

if uploaded_file:
    if 'df' not in st.session_state:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.session_state.df = df.copy()
    else:
        df = st.session_state.df

    st.success('✅ File uploaded successfully!')
    st.subheader('📌 Original Data Preview:')
    st.dataframe(df, use_container_width=True)

    if st.button('🔎 Generate AI Cleaning Suggestions'):
        with st.spinner('Generating AI suggestions...'):
            prompt_cleaning = f"Dataset:\n{df.head().to_string()}\nProvide concise bullet points for data cleaning steps clearly."

            completion_cleaning = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt_cleaning}]
            )

            suggestions = completion_cleaning.choices[0].message.content
            st.session_state.suggestions = [s.strip('- ') for s in suggestions.strip().split('\n') if s]

    if st.session_state.suggestions:
        st.subheader('🧹 AI Cleaning Suggestions:')
        selected_steps = []
        for step in st.session_state.suggestions:
            if st.checkbox(step, value=True):
                selected_steps.append(step)

        if st.button('✅ Apply Cleaning Steps'):
            cleaned_df = df.copy()
            actions_applied = []

            for step in selected_steps:
                step_lower = step.lower()
                # Drop columns
                if 'drop' in step_lower and 'column' in step_lower:
                    col_to_drop = step.split("'")[1]
                    if col_to_drop in cleaned_df.columns:
                        cleaned_df.drop(columns=[col_to_drop], inplace=True)
                        actions_applied.append(f"Dropped column '{col_to_drop}'.")

                # Fill missing values with median
                elif 'fill missing values' in step_lower:
                    numeric_cols = cleaned_df.select_dtypes(include=['float64', 'int64']).columns
                    for col in numeric_cols:
                        median = cleaned_df[col].median()
                        cleaned_df[col].fillna(median, inplace=True)
                    actions_applied.append("Filled missing numerical values clearly with median.")

                # One-hot encoding
                elif 'one-hot encoding' in step_lower:
                    cat_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns
                    cleaned_df = pd.get_dummies(cleaned_df, columns=cat_cols, drop_first=True)
                    actions_applied.append("Applied one-hot encoding to categorical variables.")

                # Remove duplicates
                elif 'remove duplicates' in step_lower:
                    before = len(cleaned_df)
                    cleaned_df.drop_duplicates(inplace=True)
                    removed = before - len(cleaned_df)
                    actions_applied.append(f"Removed {removed} duplicate rows.")

                # Standardize numerical columns
                elif 'standardize numerical columns' in step_lower:
                    numeric_cols = cleaned_df.select_dtypes(include=['float64', 'int64']).columns
                    cleaned_df[numeric_cols] = (cleaned_df[numeric_cols] - cleaned_df[numeric_cols].mean()) / cleaned_df[numeric_cols].std()
                    actions_applied.append("Standardized numerical columns.")

            st.success("✅ Cleaning applied successfully!")

            st.subheader("📝 Applied Actions Clearly:")
            for action in actions_applied:
                st.info(action)

            st.subheader("✨ Cleaned Data Preview:")
            st.dataframe(cleaned_df.head(), use_container_width=True)

            buffer = io.BytesIO()
            cleaned_df.to_csv(buffer, index=False)
            buffer.seek(0)

            st.download_button(
                label="⬇️ Download Cleaned Data",
                data=buffer,
                file_name="cleaned_data.csv",
                mime="text/csv"
            )

            # Insights & Visualization
            with st.spinner('Generating insights clearly...'):
                prompt_insights = f"Summarize key insights clearly from:\n{cleaned_df.head().to_string()}"

                completion_insights = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt_insights}]
                )

                insights = completion_insights.choices[0].message.content
                st.subheader("📈 AI Insights (Clearly):")
                st.write(insights)

                numeric_columns = cleaned_df.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_columns) >= 2:
                    fig = px.scatter(cleaned_df, x=numeric_columns[0], y=numeric_columns[1],
                                     title=f"{numeric_columns[0]} vs {numeric_columns[1]} (AI Recommended clearly)")
                    st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("⚠️ Click 'Generate AI Cleaning Suggestions' to start clearly.")




