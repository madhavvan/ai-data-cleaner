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
    st.error("❌ OpenAI API Key not found clearly.")
    st.stop()

client = OpenAI(api_key=openai_api_key)

st.title('🚀 Professional AI Data Cleaner & Insights Generator')

uploaded_file = st.file_uploader('Upload CSV or Excel clearly:', ['csv', 'xlsx'])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success('✅ Data uploaded clearly!')
        st.dataframe(df.head())

        if st.button('Generate AI Cleaning Suggestions clearly'):
            with st.spinner('Analyzing clearly...'):
                prompt_cleaning = f"This dataset:\n{df.head().to_string()}\nBrief bullet points on cleaning steps clearly."

                completion_cleaning = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt_cleaning}]
                )

                suggestions = completion_cleaning.choices[0].message.content
                suggestions_list = [s.strip('- ') for s in suggestions.strip().split('\n') if s]

                selected_steps = []
                for step in suggestions_list:
                    if st.checkbox(step, value=True):
                        selected_steps.append(step)

                if st.button('Apply Selected Steps clearly'):
                    cleaned_df = df.copy()

                    for step in selected_steps:
                        step_lower = step.lower()
                        if 'remove column' in step_lower:
                            col_to_remove = step.split('remove column')[-1].strip().strip("'\"")
                            if col_to_remove in cleaned_df.columns:
                                cleaned_df.drop(columns=col_to_remove, inplace=True)
                        elif 'fill missing values' in step_lower:
                            cleaned_df.fillna(method='ffill', inplace=True)
                        elif 'remove duplicates' in step_lower or 'drop duplicates' in step_lower:
                            cleaned_df.drop_duplicates(inplace=True)
                        elif 'remove rows with missing' in step_lower:
                            cleaned_df.dropna(inplace=True)

                    st.success("✅ Data cleaned automatically clearly!")
                    st.dataframe(cleaned_df.head())

                    buffer = io.BytesIO()
                    cleaned_df.to_csv(buffer, index=False)
                    buffer.seek(0)

                    st.download_button(
                        label="Download Cleaned Data clearly",
                        data=buffer,
                        file_name="cleaned_data.csv",
                        mime="text/csv"
                    )

                    # Generate insights and visualizations clearly
                    with st.spinner('Generating AI Insights clearly...'):
                        prompt_insights = f"Briefly summarize clearly key insights and suggest one or two charts (clearly) from this cleaned dataset:\n{cleaned_df.head().to_string()}"

                        completion_insights = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt_insights}]
                        )

                        insights = completion_insights.choices[0].message.content
                        st.subheader("🔍 AI Insights clearly:")
                        st.write(insights)

                        # Example visualization (automatic based on AI clearly)
                        numeric_columns = cleaned_df.select_dtypes(include=['int64', 'float64']).columns
                        if len(numeric_columns) >= 2:
                            fig = px.scatter(cleaned_df, x=numeric_columns[0], y=numeric_columns[1],
                                             title=f"{numeric_columns[0]} vs {numeric_columns[1]} (AI Recommended clearly)")
                            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error clearly: {e}")
