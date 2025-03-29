import streamlit as st
import pandas as pd
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

st.set_page_config(page_title="AI Data Cleaner & Analyzer", layout="wide")

st.title('🚀 AI Data Cleaner & Analyzer')

uploaded_file = st.file_uploader('Upload your CSV or Excel:', ['csv', 'xlsx'])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success('✅ File uploaded successfully!')

        st.subheader('📌 Original Data Preview:')
        st.dataframe(df, use_container_width=True)

        if st.button('🔎 Generate AI Cleaning Suggestions'):
            with st.spinner('AI is generating suggestions...'):
                prompt_cleaning = f"Here's the dataset:\n{df.head().to_string()}\nSuggest very clear and concise cleaning steps as bullet points (e.g., remove specific columns, fill missing values, drop duplicates, etc.)."

                completion_cleaning = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt_cleaning}]
                )

                suggestions = completion_cleaning.choices[0].message.content
                suggestions_list = [s.strip('- ') for s in suggestions.strip().split('\n') if s]

                st.subheader('🧹 AI Cleaning Suggestions:')
                selected_steps = []
                for step in suggestions_list:
                    if st.checkbox(step, value=True):
                        selected_steps.append(step)

                if st.button('✅ Apply Cleaning Steps'):
                    cleaned_df = df.copy()
                    actions_applied = []

                    for step in selected_steps:
                        step_lower = step.lower()
                        # Remove columns
                        if 'remove column' in step_lower:
                            col_to_remove = step.split('remove column')[-1].strip().strip("'\"")
                            if col_to_remove in cleaned_df.columns:
                                cleaned_df.drop(columns=col_to_remove, inplace=True)
                                actions_applied.append(f"Removed column '{col_to_remove}' clearly.")
                        # Fill missing values
                        elif 'fill missing values' in step_lower:
                            cleaned_df.fillna(method='ffill', inplace=True)
                            actions_applied.append("Filled missing values clearly using forward-fill.")
                        # Remove duplicates
                        elif 'remove duplicates' in step_lower or 'drop duplicates' in step_lower:
                            count_before = len(cleaned_df)
                            cleaned_df.drop_duplicates(inplace=True)
                            removed = count_before - len(cleaned_df)
                            actions_applied.append(f"Removed {removed} duplicates clearly.")
                        # Remove rows with missing values
                        elif 'remove rows with missing' in step_lower:
                            count_before = len(cleaned_df)
                            cleaned_df.dropna(inplace=True)
                            removed = count_before - len(cleaned_df)
                            actions_applied.append(f"Removed {removed} rows with missing values clearly.")

                    st.success("✅ Cleaning applied successfully!")

                    st.subheader("📝 Actions Applied Clearly:")
                    for action in actions_applied:
                        st.info(action)

                    st.subheader("✨ Cleaned Data Preview:")
                    st.dataframe(cleaned_df, use_container_width=True)

                    buffer = io.BytesIO()
                    cleaned_df.to_csv(buffer, index=False)
                    buffer.seek(0)

                    st.download_button(
                        label="⬇️ Download Cleaned Data",
                        data=buffer,
                        file_name="cleaned_data.csv",
                        mime="text/csv"
                    )

                    # Insights and Visualization
                    with st.spinner('Generating Insights & Visualization clearly...'):
                        prompt_insights = f"Briefly summarize key insights of this dataset clearly:\n{cleaned_df.head().to_string()}"

                        completion_insights = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt_insights}]
                        )

                        insights = completion_insights.choices[0].message.content
                        st.subheader("📈 AI-generated Insights Clearly:")
                        st.write(insights)

                        numeric_columns = cleaned_df.select_dtypes(include=['int64', 'float64']).columns
                        if len(numeric_columns) >= 2:
                            fig = px.scatter(cleaned_df, x=numeric_columns[0], y=numeric_columns[1],
                                             title=f"{numeric_columns[0]} vs {numeric_columns[1]} clearly recommended by AI")
                            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error clearly: {e}")
