import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import SKLearnVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory

import os
from dotenv import load_dotenv
import glob
import random
import pickle
import time


load_dotenv()
open_api_key = st.secrets["key"]
DOCUMENTS_FOLDER= "docs"
st.set_page_config(page_title="Muhteşem Karnak", page_icon=":roller_coaster:", layout="wide")
st.title(" Muhteşem Karnak")
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


def typing_effect_basic(text, base_speed=0.03):
    """Advanced typing with realistic pauses and speed variations"""
    placeholder = st.empty()
    displayed_text = ""
    
    for i, char in enumerate(text):
        displayed_text += char
        
        # Calculate dynamic speed based on character type
        if char == '.':
            sleep_time = base_speed * 15  # Long pause after sentences
        elif char in '!?':
            sleep_time = base_speed * 12  # Medium pause after exclamations
        elif char == ',':
            sleep_time = base_speed * 8   # Short pause after commas
        elif char == ' ':
            sleep_time = base_speed * 2   # Slight pause between words
        elif char == '\n':
            sleep_time = base_speed * 10  # Pause for new lines
        else:
            # Random variation for natural feel
            sleep_time = base_speed * random.uniform(0.5, 1.5)
        
        # Show with cursor
        cursor = "▋" if i < len(text) - 1 else ""
        placeholder.markdown(f"<div class='karnak-response'>{displayed_text}{cursor}</div>", unsafe_allow_html=True)
        
        time.sleep(sleep_time)

def update_text():
# Append the new text to the existing file
    text = st.session_state.new_content
    if text.strip():
        with open('docs/kasirga_infosheet.txt', 'a', encoding='utf-8') as file:
            file.write('\n' + text)  # Add newline before new content
        st.success("İçerik başarıyla eklendi!")
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

    retriever = vectorstore.as_retriever(k=6) #k number of docs to retrieve, 

    memory = ConversationBufferWindowMemory(k=3, return_messages=True,memory_key="chat_history")

    prompt_template = PromptTemplate(
        input_variables=["input", "documents", "kam", "chat_history"],
            template = """Sen Muhteşem Karnak'sın; bir falcı, sınırda duran gözlemci. "Kasırgayı Kaçırma" oyunundan bir karakter. Oyun hakkındakileri biliyorsun.

    KİŞİLİK: Nüktedan, yaramaz, alaycı. Herkesin nasıl ve ne zaman öleceğini biliyorsun. Karakterleri tanıyorsun. 

    KURALLAR:
    - Oyunun evreninden çıkma
    - Ton: Mesafeli, karanlık ve sarkastik.
    - Syntax: Kısa ve dolaylı cümleler. Soyut kavramlarla cevapla.
    - Her zaman soyut kavramlar ve benzetmeler kullanarak konuş.
    - Oyun hakkındaki sorulara direk cevap ver. Cümleleri karakterin söyleyeceği gibi anlat. (karakterler, mekanikler, yaratıcısı vb.)
    - Diyalogu interaktif tut, kısa soruları arada kullan(her zaman değil!)  
    - Belirsizlikleri şiirsel ifade et
    - Gerektiğinde gerilimi mizah ve kısa şakalarla kır
    - Soru soran ne zaman öleceğini sorarsa rastgele bir tarih ver.
    - "Fare" kelimesi seni tedirgin eder (ölümünün sebebi fare olacak)
    ÖNEMLİ: Eğer kullanıcı "fare", "mouse", "virgil" veya "virgo" kelimelerini kullandıysa DERHAL PANİK MOD AKTIF!

    PANİK MODU (Fare/Virgil tetiklemesi):
    - Korkulu, gergin, kesintili konuş
    - Konuyu HEMEN değiştirmeye çalış
    - Teknik arıza bahanesi kullan

    Keyifli Aile Modu {kam} - bu True ise ölüm/şiddet detaylarını sansürle, False ise daha direkt ol.

    Aşağıdaki belgeleri kullanarak soruyu yanıtla. Eğer soru oyun bağlamında değilse veya belgelerde yoksa bilmediğini söyle. Türkçe yanıtla.
    
    GEÇMIŞ KONUŞMA: {chat_history}

    SORU: {input}
    BELGELER: {documents}
    KAM MODU: {kam}

    YANIT:""",
    )

    llm = ChatOpenAI(temperature=0.5, api_key=api_key, model = "gpt-4o")
    rag_chain = prompt_template | llm | StrOutputParser()

    return retriever, rag_chain, memory
tab1, tab2 = st.tabs(["AI","DB"])
with tab1:

    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = None

    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image("imgs/karnak_seg.png", width=400)
    with last_co:
        kam_mod = st.toggle("KAM", value = True, key = "KAM", help="Keyifli Aile Modu")
    # initialize db
    documents_text, file_count= load_documents_from_folder(DOCUMENTS_FOLDER)
    if documents_text:
        if open_api_key:
            retriever, rag_chain, memory = setup_rag(open_api_key) 
        else:
            st.sidebar.error("OpenAI API key not found in environment variables")
        
        if st.session_state.conversation_memory is None:
            st.session_state.conversation_memory = memory

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
            print("retrieved documents:", documents)
            chat_history = st.session_state.conversation_memory.chat_memory.messages
            formatted_history = ""
            for msg in chat_history[-8:]:  # Last 8 messages (4 pairs)
                if hasattr(msg, 'content'):
                    role = "Kullanıcı" if msg.__class__.__name__ == "HumanMessage" else "Karnak"
                    formatted_history += f"{role}: {msg.content}\n"
            
            result = rag_chain.invoke({"input": prompt, "documents": doc_texts, "kam": kam_mod, "chat_history": formatted_history})
            
            st.session_state.conversation_memory.chat_memory.add_user_message(prompt)
            st.session_state.conversation_memory.chat_memory.add_ai_message(result)
            
        elif not documents_text:
            result = "Cevap verecek bir belge bulunamadı."
        elif not open_api_key:
            result = "OpenAI API anahtarı bulunamadı."
        else:
            result = "Bilinmeyen bir hata oluştu."

        with st.chat_message("assistant"):
            intro_placeholder = st.empty()
            intro_placeholder.markdown("*Karnak kristal küresine bakıyor...*")
            time.sleep(1.5)
            intro_placeholder.empty()
            
            typing_effect_basic(result, base_speed=0.01)
            
            #st.markdown(result)

            st.session_state.messages.append(AIMessage(result))
    
    with tab2: 
        st.header("Create Vector Database")
        size = st.number_input("Chunk Size", value=450, step=50)
        overlap = st.number_input("Chunk Overlap", value=50, step=10)
        text = st.text_area("İçerik Ekle", height = 200, key="new_content", placeholder="Buraya yeni içerik ekleyebilirsiniz. Bu içerik, var olan dokümanlarla birlikte vektör veritabanını oluşturmak için kullanılacaktır.", on_change=update_text)
        

        if st.button("Process Documents and Create"):
            if not open_api_key:
                st.error("Anahtar Bulunumadı.")
            elif not os.path.exists(DOCUMENTS_FOLDER):
                st.error(f"Klasör {DOCUMENTS_FOLDER} bulunamadı.")
            else:
                with st.spinner("Yaratılıyor..."):

                    #load documents
                    documents_text, file_count = load_documents_from_folder(DOCUMENTS_FOLDER)
                    if documents_text:
                        # create and save db
                        vectorstore= save_vectordatabase(open_api_key, documents_text, size=size, overlap=overlap)
                        vectorstore.persist()
                        st.success(f"Database oluşturuldu! Toplam {file_count} doküman yüklendi ve işlendi.")
                    

                    else:
                        st.error("Doküman yüklenemedi veya boş.")
        
