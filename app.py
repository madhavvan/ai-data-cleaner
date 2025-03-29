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

st.set_page_config(page_title="AI Data Cleaner", layout="wide")

st.title('🚀 Robust AI Data Cleaner')

uploaded_file = st.file_uploader('Upload CSV or Excel:', ['csv', 'xlsx'])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    st.success('✅ File uploaded successfully!')
    st.dataframe(df.head())

    if st.button('Generate AI Cleaning Suggestions'):
        with st.spinner('Generating suggestions clearly...'):
            prompt_cleaning = f"Dataset preview:\n{df.head().to_string()}\nClearly specify bullet-point data cleaning steps (remove columns, fill missing values, remove invalid rows, one-hot encoding)."

            completion_cleaning = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt_cleaning}]
            )

            suggestions = completion_cleaning.choices[0].message.content
            st.session_state.suggestions = suggestions.split('\n')

    if 'suggestions' in st.session_state:
        st.subheader('🧹 AI Cleaning Suggestions:')
        selected_steps = []
        for step in st.session_state.suggestions:
            step = step.strip('- ')
            if step and st.checkbox(step, True):
                selected_steps.append(step)

        if st.button('Apply Cleaning Steps'):
            cleaned_df = df.copy()
            actions_applied = []

            for step in selected_steps:
                step_lower = step.lower()

                # Remove column explicitly
                if 'remove column' in step_lower or 'drop column' in step_lower:
                    column_name = step.split("'")[1].strip()
                    if column_name in cleaned_df.columns:
                        cleaned_df.drop(columns=column_name, inplace=True)
                        actions_applied.append(f"Column '{column_name}' removed clearly.")

                # Fill missing numeric values
                elif 'fill missing values' in step_lower:
                    numeric_cols = cleaned_df.select_dtypes(include=['float64', 'int64']).columns
                    for col in numeric_cols:
                        median = cleaned_df[col].median()
                        cleaned_df[col].fillna(median, inplace=True)
                    actions_applied.append("Missing numeric values filled clearly with median.")

                # Remove rows with null values
                elif 'remove rows with missing' in step_lower or 'remove rows with null' in step_lower:
                    before_rows = len(cleaned_df)
                    cleaned_df.dropna(inplace=True)
                    removed_rows = before_rows - len(cleaned_df)
                    actions_applied.append(f"Removed {removed_rows} rows with missing values clearly.")

                # Remove rows with special/invalid characters explicitly
                elif 'remove rows with invalid' in step_lower or 'special characters' in step_lower:
                    cols_with_special = cleaned_df.select_dtypes(include=['object']).columns
                    before_rows = len(cleaned_df)
                    for col in cols_with_special:
                        cleaned_df = cleaned_df[cleaned_df[col].str.match("^[a-zA-Z0-9_ ]*$", na=False)]
                    removed_rows = before_rows - len(cleaned_df)
                    actions_applied.append(f"Removed {removed_rows} rows containing invalid/special characters clearly.")

                # One-hot encoding categorical variables
                elif 'one-hot encoding' in step_lower:
                    cat_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns
                    cleaned_df = pd.get_dummies(cleaned_df, columns=cat_cols, drop_first=True)
                    actions_applied.append("Categorical columns encoded with one-hot encoding clearly.")

            st.success("✅ Data cleaned with actual changes clearly!")

            st.subheader("📝 Actions Applied Clearly:")
            for action in actions_applied:
                st.info(action)

            st.subheader("✨ Fully Cleaned Data:")
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
                prompt_insights = f"Summarize key insights clearly from dataset:\n{cleaned_df.head().to_string()}"

                completion_insights = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt_insights}]
                )

                insights = completion_insights.choices[0].message.content
                st.subheader("📈 AI-generated Insights (Clearly):")
                st.write(insights)

                numeric_columns = cleaned_df.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_columns) >= 2:
                    fig = px.scatter(cleaned_df, x=numeric_columns[0], y=numeric_columns[1],
                                     title=f"{numeric_columns[0]} vs {numeric_columns[1]} (AI Recommended clearly)")
                    st.plotly_chart(fig, use_container_width=True)