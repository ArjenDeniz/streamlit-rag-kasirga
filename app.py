import streamlit as st
from langchain_openai import ChatOpenAI

st.title(" Quick Rag App")

open_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

def generate_response(input_text):
    model = ChatOpenAI(temperature=0.6, api_key = open_api_key)
    st.info(model.invoke(input_text))

with st.form("xx"):
    text = st.text_area(
        "Enter text:",
        "Tell me a joke"
    )
    submitted = st.form_submit_button("Submit")

    if not open_api_key.startswith("sk-"):
        st.warning("please enter your api key")
    if submitted and open_api_key.startswith("sk-"):
        generate_response(text)