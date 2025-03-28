import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI

# load API key clearly
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

st.title('🧠 AI Data Cleaner & Insights Generator')

uploaded_file = st.file_uploader('Upload CSV/Excel:', type=['csv', 'xlsx'])

if uploaded_file:
    try:
        # Load data
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success('✅ File uploaded clearly!')
        st.dataframe(df.head())

        if st.button('Generate AI Cleaning Suggestions'):
            with st.spinner('AI clearly analyzing your data...'):
                prompt = f"This is the dataset:\n{df.head().to_string()}\nSuggest steps to clean this data clearly."

                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )

                suggestions = completion.choices[0].message.content

                st.subheader('AI-generated Suggestions clearly:')
                st.write(suggestions)
