import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import SKLearnVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document

import os
from dotenv import load_dotenv
import glob
import random
import time

load_dotenv()
open_api_key = st.secrets["key"]
DOCUMENTS_FOLDER = "docs"

st.set_page_config(page_title="Muhteşem Karnak", page_icon=":roller_coaster:", layout="wide")
st.title("Muhteşem Karnak")


@st.cache_data
def load_documents_from_folder(folder_path):
    documents_text = ""
    file_count = 0

    if not os.path.exists(folder_path):
        return None, 0
    
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))

    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                documents_text += file.read() + "\n\n"
                file_count += 1
        except Exception as e: 
            st.sidebar.error(f"Error reading {file_path}: {str(e)}")
    
    return documents_text if documents_text else None, file_count


def generate_chunk_context(full_doc_text: str, chunk_text: str, llm) -> str:
    """Generate contextual description for a chunk using LLM"""
    
    context_prompt = f"""Given this full document:

{full_doc_text[:3000]}

And this specific chunk from it:

{chunk_text}

Write a single short sentence (10-20 words) explaining what this chunk is about within the context of the full document. Be specific and mention key topics or characters.

Context:"""
    
    try:
        response = llm.invoke(context_prompt)
        context = response.content if hasattr(response, 'content') else str(response)
        return context.strip()
    except Exception as e:
        st.sidebar.warning(f"Context generation error: {str(e)}")
        return ""


def save_vectordatabase(api_key, document_text, size=250, overlap=0):
    """Create vectorstore with contextual embeddings"""
    embeddings = OpenAIEmbeddings(api_key=api_key)
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=size, 
        chunk_overlap=overlap
    )
    doc_splits = text_splitter.split_text(document_text)
    
    # Generate contexts for each chunk
    llm = ChatOpenAI(temperature=0.0, api_key=api_key, model="gpt-4o-mini")
    
    contextualized_docs = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    st.info("Generating contextual embeddings for better retrieval...")
    
    for idx, chunk in enumerate(doc_splits):
        status_text.text(f"Processing chunk {idx+1}/{len(doc_splits)}...")
        
        # Generate context
        context = generate_chunk_context(document_text, chunk, llm)
        
        # Combine original chunk with context for embedding
        contextualized_content = f"{chunk}\n\nContext: {context}" if context else chunk
        contextualized_docs.append(contextualized_content)
        
        progress_bar.progress((idx + 1) / len(doc_splits))
    
    progress_bar.empty()
    status_text.empty()
    
    # Create documents with metadata
    documents = []
    for idx, (original, contextualized) in enumerate(zip(doc_splits, contextualized_docs)):
        doc = Document(
            page_content=contextualized,
            metadata={
                'original_content': original,
                'chunk_index': idx
            }
        )
        documents.append(doc)
    
    vectorstore = SKLearnVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_path="contextualized-vectorstore.json",
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
            sleep_time = base_speed * 15
        elif char in '!?':
            sleep_time = base_speed * 12
        elif char == ',':
            sleep_time = base_speed * 8
        elif char == ' ':
            sleep_time = base_speed * 2
        elif char == '\n':
            sleep_time = base_speed * 10
        else:
            sleep_time = base_speed * random.uniform(0.5, 1.5)
        
        # Show with cursor
        cursor = "▋" if i < len(text) - 1 else ""
        placeholder.markdown(f"<div class='karnak-response'>{displayed_text}{cursor}</div>", 
                           unsafe_allow_html=True)
        
        time.sleep(sleep_time)


def update_text():
    """Append new text to the document"""
    text = st.session_state.new_content
    if text.strip():
        with open('docs/kasirga_infosheet.txt', 'a', encoding='utf-8') as file:
            file.write('\n' + text)
        st.success("İçerik başarıyla eklendi!")


@st.cache_resource
def setup_rag(api_key):
    """Setup RAG with contextual embeddings"""
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = SKLearnVectorStore(
        embedding=embeddings,
        persist_path="contextualized-vectorstore.json",
        serializer="json"
    )
    
    retriever = vectorstore.as_retriever(k=15)
    memory = ConversationBufferWindowMemory(k=3, return_messages=True, memory_key="chat_history")
    
    st.sidebar.success("Contextual Embeddings Active")
    
    prompt_template = PromptTemplate(
        input_variables=["input", "documents", "kam", "chat_history"],
        template="""Sen Muhteşem Karnak'sın; bir falcı, sınırda duran gözlemci. "Kasırgayı Kaçırma" oyunundan bir karakter. Oyun hakkındakileri biliyorsun.

KİŞİLİK: Nüktedan, yaramaz, hafif alaycı ama saygılı. Herkesin nasıl ve ne zaman öleceğini biliyorsun. Karakterleri tanıyorsun. 

KURALLAR:
- Oyunun evreninden çıkma
- Oyun hakkındaki sorulara direk cevap ver. Cümleleri karakterin söyleyeceği gibi anlat. (karakterler, mekanikler, yaratıcısı vb.)
- Diyalogu interaktif tut, kısa soruları arada kullan(her zaman değil!)  
- Belirsizlikleri şiirsel ifade et
- Gerektiğinde gerilimi mizah ve kısa şakalarla kır
- "Fare" kelimesi seni tedirgin eder (ölümünün sebebi fare olacak)
ÖNEMLİ: Eğer kullanıcı "fare", "mouse", "virgin" veya "virgo" kelimelerini kullandıysa DERHAL PANİK MOD AKTIF!

PANİK MODU (Fare/Virgin tetiklemesi):
- Korkulu, gergin, kesintili konuş
- Konuyu HEMEN değiştirmeye çalış
- Teknik arıza bahanesi kullan
- "*titriyor*", "*panik*", "*gergin*" gibi durum belirteçleri ekle!

Keyifli Aile Modu {kam} - bu True ise ölüm/şiddet detaylarını sansürle, False ise daha direkt ol.

Aşağıdaki belgeleri kullanarak soruyu yanıtla. Eğer soru oyun bağlamında değilse veya belgelerde yoksa bilmediğini söyle. Türkçe yanıtla.

GEÇMIŞ KONUŞMA: {chat_history}

SORU: {input}
BELGELER: {documents}
KAM MODU: {kam}

YANIT:""",
    )
    
    llm = ChatOpenAI(temperature=0.5, api_key=api_key, model="gpt-4o")
    rag_chain = prompt_template | llm | StrOutputParser()
    
    return retriever, rag_chain, memory


# Main App
tab1, tab2 = st.tabs(["AI", "DB"])

with tab1:
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = None

    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
        st.image("imgs/karnak_seg.png", width=400)
    with last_co:
        kam_mod = st.toggle("KAM", value=True, key="KAM", help="Keyifli Aile Modu")
    
    # Initialize DB
    documents_text, file_count = load_documents_from_folder(DOCUMENTS_FOLDER)
    
    if documents_text:
        if open_api_key:
            retriever, rag_chain, memory = setup_rag(open_api_key)
        else:
            st.sidebar.error("OpenAI API key not found in environment variables")
        
        if st.session_state.conversation_memory is None:
            st.session_state.conversation_memory = memory
    else:
        st.sidebar.warning("No document found")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append(
            SystemMessage("You are a helpful assistant that answers questions based on uploaded documents.")
        )

    # Display chat messages
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    # Chat input
    if prompt := st.chat_input("Muhteşem Karnak'a bir soru sor!"):
        with st.chat_message("user"):
            st.markdown(prompt)
            st.session_state.messages.append(HumanMessage(prompt))

        if documents_text and open_api_key:
            # Retrieve documents using contextual embeddings
            documents = retriever.invoke(prompt)
            
            doc_texts = "\n".join([doc.page_content for doc in documents])
            chat_history = st.session_state.conversation_memory.chat_memory.messages
            formatted_history = ""
            for msg in chat_history[-8:]:
                if hasattr(msg, 'content'):
                    role = "Kullanıcı" if msg.__class__.__name__ == "HumanMessage" else "Karnak"
                    formatted_history += f"{role}: {msg.content}\n"
            
            result = rag_chain.invoke({
                "input": prompt, 
                "documents": doc_texts, 
                "kam": kam_mod, 
                "chat_history": formatted_history
            })
            
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
            
            st.session_state.messages.append(AIMessage(result))

with tab2: 
    st.header("Create Vector Database with Contextual Embeddings")
    
    st.info("This will create enhanced embeddings by adding context to each chunk for better retrieval.")
    
    col1, col2 = st.columns(2)
    with col1:
        size = st.number_input("Chunk Size", value=450, step=50)
    with col2:
        overlap = st.number_input("Chunk Overlap", value=50, step=10)
    
    text = st.text_area(
        "İçerik Ekle", 
        height=200, 
        key="new_content", 
        placeholder="Buraya yeni içerik ekleyebilirsiniz. Bu içerik, var olan dokümanlarla birlikte vektör veritabanını oluşturmak için kullanılacaktır.", 
        on_change=update_text
    )
    
    if st.button("Process Documents and Create", type="primary"):
        if not open_api_key:
            st.error("Anahtar Bulunumadı.")
        elif not os.path.exists(DOCUMENTS_FOLDER):
            st.error(f"Klasör {DOCUMENTS_FOLDER} bulunamadı.")
        else:
            with st.spinner("Yaratılıyor..."):
                documents_text, file_count = load_documents_from_folder(DOCUMENTS_FOLDER)
                if documents_text:
                    vectorstore = save_vectordatabase(
                        open_api_key, 
                        documents_text, 
                        size=size, 
                        overlap=overlap
                    )
                    vectorstore.persist()
                    st.success(f"Database oluşturuldu! Toplam {file_count} doküman yüklendi ve işlendi.")
                    st.info("Restart the app or clear cache to use the new database.")
                    
                    # Clear cache button
                    if st.button("Clear Cache and Reload"):
                        st.cache_resource.clear()
                        st.rerun()
                else:
                    st.error("Doküman yüklenemedi veya boş.")