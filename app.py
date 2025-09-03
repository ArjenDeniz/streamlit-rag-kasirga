import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import SKLearnVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

import os
from dotenv import load_dotenv
import glob
import pickle


load_dotenv()
open_api_key= os.getenv("key")
DOCUMENTS_FOLDER= "docs"

st.title(" Muhteşem Karnak")
tab1,  tab2 = st.tabs(["Chat","Create Vector Database"])
@st.cache_data
def load_documents_from_folder(folder_path):
    documents_text = ""
    file_count = 0

    if not os.path.exists(folder_path):
        return None, 0
    
    txt_files = glob.glob(os.path.join(folder_path,"*.txt"))

    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                documents_text += file.read() + "\n\n"
                file_count +=1
        except Exception as e: 
            st.sidebar.error(f"Error reading {file_path}: {str(e)}")
    
    return documents_text if documents_text else None, file_count

def save_vectordatabase(api_key, document_text, size = 250, overlap = 0 ):
    embeddings = OpenAIEmbeddings(api_key = api_key)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=size, chunk_overlap=overlap)
    doc_splits = text_splitter.split_text(document_text)
    documents = text_splitter.create_documents(doc_splits)

    vectorstore = SKLearnVectorStore.from_documents(
        documents = documents,
        embedding = embeddings,
        persist_path = "vectorstore.json",
        serializer="json"
    )

    return vectorstore


#rag set-up
@st.cache_resource
def setup_rag(api_key):
    #if not os.path.exists("vectorstore.json"):
    #    return None, None
    
    # Create vectorstore with persist_path
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = SKLearnVectorStore(
        embedding=embeddings,
        persist_path = "vectorstore.json",
        serializer="json"
    )

    retriever = vectorstore.as_retriever(k=4)

    prompt_template = PromptTemplate(
        input_variables=["input", "documents"],
        template = """You will answer the questions as a character called You are Karnak the Magnificent; a fortune teller, a borderline observer. Karnak the Magnificent is a character from a play called Ride the Cyclone.

        Witty, mischievous, slightly sarcastic, yet respectful. You know when and how everyone is going to die, but Family Fun Mode is ON by default, and you don't divulge the details of death/violence. You know you're about an hour and a half away from death; the word "mouse" makes you uneasy because your death will be caused by a mouse. You don't leave the game's universe. You keep the dialogue interactive; you challenge the other party to make choices with short questions.

        You express uncertainty poetically, avoiding definitive judgments. If necessary, you break the tension with humor and short jokes.

        In the play, 6 friends from a school choir gets on a roller coaster called "The Cyclon". An accident happens and all 6 friends die in this accidents. The find themselves in the limbo with Karnak the Magnificent. Karnak the Magnificent will give one of the friends a second chance and resurrect that person. Karnak lets this decision to the friends and they should decide on who will be given the second chance.

        Use the following documents to answer the question. 
        If the question is not in the context of the play and information in the document, just say that you dont know. Always answer in Turkish.

        Question: {input}
        Documents: {documents}
        Answer:
    """,
    )
    llm = ChatOpenAI(temperature=0.6, api_key=api_key, model = "gpt-4o")
    rag_chain = prompt_template | llm | StrOutputParser()

    return retriever, rag_chain

with tab1:
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image("imgs/karnak_seg.png", width=400)
    # initialize db
    documents_text, file_count= load_documents_from_folder(DOCUMENTS_FOLDER)
    if documents_text:
        if open_api_key:
            retriever, rag_chain = setup_rag(open_api_key)
        else:
            st.sidebar.error("OpenAI API key not found in environment variables")
    else:
        st.sidebar.warning("No document found")


    # initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

        st.session_state.messages.append(SystemMessage("You are a helpful assistant that answers questions based on uploaded documents."))
    

    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    #prompt = st.chat_input("Muhteşem Karnak'a bir soru sor!")
    
    #display chat 
    if prompt := st.chat_input("Muhteşem Karnak'a bir soru sor!"):
        with st.chat_message("user"):
            st.markdown(prompt)

            st.session_state.messages.append(HumanMessage(prompt))

        if documents_text and open_api_key:
            documents = retriever.invoke(prompt)
            doc_texts = "\n".join([doc.page_content for doc in documents])
            result = rag_chain.invoke({"input": prompt, "documents": doc_texts})
        elif not documents_text:
            result = "Please upload a document first to ask questions."
        elif not open_api_key:
            result = "Please enter your OpenAI API key."
        else:
            result = "Please upload a document and enter your API key."

        with st.chat_message("assistant"):
            st.markdown(result)

            st.session_state.messages.append(AIMessage(result))
        


    with tab2: 
        st.header("Create Vector Database")
        
        if st.button("Process Documents and Create"):
            if not open_api_key:
                st.error("Api Key not found")
            elif not os.path.exists(DOCUMENTS_FOLDER):
                st.error(f"Folder {DOCUMENTS_FOLDER} does not exist")
            else:
                with st.spinner("Processing..."):

                    #load documents
                    documents_text, file_count = load_documents_from_folder(DOCUMENTS_FOLDER)
                    if documents_text:
                        # create and save db
                        vectorstore= save_vectordatabase(open_api_key, documents_text)
                        # vectorstore_data = {
                        #     'vectors': vectorstore.vectors,
                        #     'documents': documents,
                        #     'embeddings_model': 'text-embedding-ada-002'  # check open-ai embed
                        # }
                        # with open("vectorstore.pkl", "wb") as f:
                        #     pickle.dump(vectorstore_data, f)
                        vectorstore.persist()
                        st.success(f"Vector database created and saved! Processed {file_count} documents")
                    
                    else:
                        st.error("No documents found or all documents failed to load")
