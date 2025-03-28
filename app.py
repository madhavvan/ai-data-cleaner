import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables clearly
load_dotenv()

st.title('🧠 AI Data Cleaner & Insights Generator')
st.write('Clearly upload your Excel or CSV file to begin.')

uploaded_file = st.file_uploader('Choose Excel or CSV clearly:', type=['csv', 'xlsx'])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success('✅ File uploaded clearly!')
        st.write('**Preview your data clearly:**')
        st.dataframe(df.head())

    except Exception as e:
        st.error(f"Error clearly reading file: {e}")
