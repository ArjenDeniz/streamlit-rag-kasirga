import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import SKLearnVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

import os
from dotenv import load_dotenv

load_dotenv()
open_api_key= os.getenv("key")

st.title(" Quick Rag App")

uploaded_file = st.sidebar.file_uploader("Upload your document", type=["txt"])

#rag set-up
@st.cache_resource
def setup_rag(document_text, api_key):
    embeddings = OpenAIEmbeddings(api_key = api_key)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
    doc_splits = text_splitter.split_text(document_text)
    documents = text_splitter.create_documents(doc_splits)

    vectorstore = SKLearnVectorStore.from_documents(
        documents = documents,
        embedding = embeddings
    )
    retriever = vectorstore.as_retriever(k=4)

    prompt_template = PromptTemplate(
        input_variables=["input", "documents"],
        template = """Use the following documents to answer the question.
        If you don't know the answer, just say that you dont know.

        Question: {input}
        Documents: {documents}
        Answer:
    """,
    )
    llm = ChatOpenAI(temperature=0.6, api_key=api_key, model = "gpt-4o")
    rag_chain = prompt_template | llm | StrOutputParser()

    return retriever, rag_chain

# initialize db
if uploaded_file:
    if open_api_key:
        document_text = uploaded_file.read().decode()
        retriever, rag_chain = setup_rag(document_text, open_api_key)
        st.sidebar.success("Document processed!")
    else:
        st.sidebar.error("OpenAI API key not found in environment variables")
else:
    st.sidebar.warning("Please upload a document to enable RAG")


# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

    st.session_state.messages.append(SystemMessage("You are a helpful assistant that answers questions based on uploaded documents."))
    
    #display chat 
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

prompt = st.chat_input("Ask a question about your document")

if prompt:

    with st.chat_message("user"):
        st.markdown(prompt)

        st.session_state.messages.append(HumanMessage(prompt))

    if uploaded_file and open_api_key:
        documents = retriever.invoke(prompt)
        doc_texts = "\n".join([doc.page_content for doc in documents])
        result = rag_chain.invoke({"input": prompt, "documents": doc_texts})
    elif not uploaded_file:
        result = "Please upload a document first to ask questions."
    elif not open_api_key:
        result = "Please enter your OpenAI API key."
    else:
        result = "Please upload a document and enter your API key."

    with st.chat_message("assistant"):
        st.markdown(result)

        st.session_state.messages.append(AIMessage(result))
