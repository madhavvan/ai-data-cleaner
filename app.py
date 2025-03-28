import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import openai

# Load environment variables (locally)
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key = os.getenv("OPENAI_API_KEY")

st.title('🧠 AI Data Cleaner & Insights Generator')

uploaded_file = st.file_uploader('Upload Excel/CSV file clearly:', type=['csv', 'xlsx'])

if uploaded_file:
    try:
        # Load data
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success('✅ File uploaded clearly!')
        st.write('**Preview Data clearly:**')
        st.dataframe(df.head())

        # Ask GPT clearly for cleaning advice
        if st.button('Generate AI Cleaning Suggestions clearly'):
            with st.spinner('🔍 AI clearly analyzing your data...'):
                load_dotenv()
                openai.api_key = os.getenv('OPENAI_API_KEY')

                prompt = f"This is the dataset:\n{df.head().to_string()}\nSuggest clear steps to clean this data (e.g., handle missing values, remove unnecessary columns, fix issues):"

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )

                suggestions = response.choices[0].message.content

                st.subheader('AI-generated Data Cleaning Suggestions:')
                st.write(response.choices[0].message.content)

    except Exception as e:
        st.error(f"Error reading file clearly: {e}")
