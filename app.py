import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load API key clearly
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=openai_api_key)

st.title('🧠 Interactive AI Data Cleaner')

uploaded_file = st.file_uploader('Upload your CSV or Excel clearly:', ['csv', 'xlsx'])

if uploaded_file:
    try:
        # Load your data clearly
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success('✅ Data uploaded clearly!')
        st.dataframe(df.head())

        if st.button('Generate AI Cleaning Suggestions'):
            with st.spinner('Analyzing data clearly...'):
                prompt = f"This dataset:\n{df.head().to_string()}\nSuggest clear steps to clean this data briefly in bullet points."

                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )

                suggestions = completion.choices[0].message.content

                st.subheader('AI Suggestions clearly:')
                suggestions_list = suggestions.strip().split('\n')

                # Interactive acceptance/rejection
                st.write("Select cleaning steps to clearly apply:")
                selected_steps = []
                for step in suggestions_list:
                    if st.checkbox(step, value=True):
                        selected_steps.append(step)

                if st.button('Apply Selected Steps clearly'):
                    st.write("✅ **Selected Cleaning Steps:**")
                    st.write(selected_steps)
                    st.success("Cleaning steps applied (simulated clearly).")

    except Exception as e:
        st.error(f"Error clearly: {e}")
