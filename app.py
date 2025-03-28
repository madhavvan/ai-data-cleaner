import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load API key clearly for local and deployed app
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY') or st.secrets.get("OPENAI_API_KEY")

if openai_api_key is None:
    st.error("❌ OpenAI API Key not found clearly. Please check your .env or Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=openai_api_key)

st.title('🧠 Interactive AI Data Cleaner')

uploaded_file = st.file_uploader('Upload your CSV or Excel clearly:', ['csv', 'xlsx'])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success('✅ Data uploaded clearly!')
        st.dataframe(df.head())

        if st.button('Generate AI Cleaning Suggestions clearly'):
            with st.spinner('Clearly analyzing data...'):
                prompt = f"Here's the dataset:\n{df.head().to_string()}\nSuggest clearly the data cleaning steps briefly as bullet points."

                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )

                suggestions = completion.choices[0].message.content

                st.subheader('AI Cleaning Suggestions (clearly):')

                # Split suggestions into bullet points clearly
                suggestions_list = [s.strip('- ') for s in suggestions.strip().split('\n') if s]

                st.write("Select the cleaning steps to apply clearly:")

                # Collect user's clearly selected suggestions
                selected_steps = []
                for step in suggestions_list:
                    if st.checkbox(step, value=True):
                        selected_steps.append(step)

                # Button to clearly confirm and apply selections
                if st.button('Apply Selected Steps clearly'):
                    # Initially just simulate applying clearly
                    st.write("✅ **You selected clearly:**")
                    st.write(selected_steps)
                    st.success("Cleaning steps clearly applied (simulated). Next we'll automate actual application.")

    except Exception as e:
        st.error(f"Error clearly: {e}")
