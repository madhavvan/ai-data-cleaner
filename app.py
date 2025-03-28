import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI
import io

# Load API key clearly
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY') or st.secrets.get("OPENAI_API_KEY")

if openai_api_key is None:
    st.error("❌ OpenAI API Key not found clearly.")
    st.stop()

client = OpenAI(api_key=openai_api_key)

st.title('🧠 Automated AI Data Cleaner')

uploaded_file = st.file_uploader('Upload CSV or Excel clearly:', ['csv', 'xlsx'])

if uploaded_file:
    try:
        # Load dataset clearly
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success('✅ Data uploaded clearly!')
        st.dataframe(df.head())

        if st.button('Generate AI Cleaning Suggestions clearly'):
            with st.spinner('Analyzing data clearly...'):
                prompt = f"Here's the dataset:\n{df.head().to_string()}\nGive cleaning steps clearly as concise bullet points (like remove columns, fill missing values clearly)."

                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )

                suggestions = completion.choices[0].message.content

                st.subheader('AI Cleaning Suggestions (clearly):')
                suggestions_list = [s.strip('- ') for s in suggestions.strip().split('\n') if s]

                st.write("Clearly select steps to apply:")

                selected_steps = []
                for step in suggestions_list:
                    if st.checkbox(step, value=True):
                        selected_steps.append(step)

                if st.button('Apply Selected Steps clearly'):
                    cleaned_df = df.copy()

                    for step in selected_steps:
                        step_lower = step.lower()
                        # Remove columns clearly
                        if 'remove column' in step_lower:
                            col_to_remove = step.split('remove column')[-1].strip().strip("'\"")
                            if col_to_remove in cleaned_df.columns:
                                cleaned_df.drop(columns=col_to_remove, inplace=True)
                        # Fill missing values clearly
                        elif 'fill missing values' in step_lower:
                            cleaned_df.fillna(method='ffill', inplace=True)
                        # Drop duplicates clearly
                        elif 'remove duplicates' in step_lower or 'drop duplicates' in step_lower:
                            cleaned_df.drop_duplicates(inplace=True)
                        # Remove rows with missing values clearly
                        elif 'remove rows with missing' in step_lower:
                            cleaned_df.dropna(inplace=True)

                    st.success("✅ Data cleaning clearly applied successfully!")
                    st.write("**Cleaned Data clearly:**")
                    st.dataframe(cleaned_df.head())

                    # Allow download cleaned data clearly
                    buffer = io.BytesIO()
                    cleaned_df.to_csv(buffer, index=False)
                    buffer.seek(0)

                    st.download_button(
                        label="Download Cleaned Data clearly",
                        data=buffer,
                        file_name="cleaned_data.csv",
                        mime="text/csv"
                    )

    except Exception as e:
        st.error(f"Error clearly: {e}")
