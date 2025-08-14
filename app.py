import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

st.title(" Quick Rag App")

open_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

    st.session_state.messages.append(SystemMessage("You are a joke maker"))
    
    #display chat 
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

prompt = st.chat_input("Tell me a joke")

if prompt:

    with st.chat_message("user"):
        st.markdown(prompt)

        st.session_state.messages.append(HumanMessage(prompt))

    model = ChatOpenAI(temperature=0.6, api_key = open_api_key, model="gpt-4o")
    result = model.invoke(st.session_state.messages).content

    with st.chat_message("assistant"):
        st.markdown(result)

        st.session_state.messages.append(AIMessage(result))
