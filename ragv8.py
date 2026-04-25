import os
import tempfile
import logging
import streamlit as st
import pandas as pd
from io import StringIO

# Official Google SDK
from google import genai

# LangChain & Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

# ==========================================
# PAGE & LOGGING CONFIGURATION
# ==========================================
st.set_page_config(page_title="Gemini RAG Assistant", page_icon="🧠", layout="wide")
st.title("🧠 SriLab.AI India, HR Policy - Personal RAG Assistant")

if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = StringIO()

def log_to_ui(message):
    st.session_state.log_buffer.write(message + "\n")
    logging.info(message)

# ==========================================
# DYNAMIC MODEL FETCHING (Bulletproofed)
# ==========================================
def get_available_models(api_key):
    """Safely lists models regardless of Google SDK version changes."""
    try:
        client = genai.Client(api_key=api_key)
        models = list(client.models.list())
        
        chat_models = []
        embed_models = []
        
        for m in models:
            actions_str = ""
            if hasattr(m, 'supported_actions') and m.supported_actions:
                actions_str = str(m.supported_actions).lower()
            elif hasattr(m, 'supported_generation_methods') and m.supported_generation_methods:
                actions_str = str(m.supported_generation_methods).lower()
                
            if 'generate' in actions_str:
                chat_models.append(m.name)
            if 'embed' in actions_str:
                embed_models.append(m.name)
                
        return chat_models, embed_models
    except Exception as e:
        st.error(f"Failed to fetch models: {e}")
        return [], []

# ==========================================
# CORE RAG FUNCTIONS
# ==========================================
@st.cache_resource
def get_vector_store(api_key, embed_model):
    os.environ["GOOGLE_API_KEY"] = api_key
    
    if not embed_model.startswith("models/"):
        embed_model = f"models/{embed_model}"
        
    embeddings = GoogleGenerativeAIEmbeddings(model=embed_model)
    return Chroma(persist_directory="./chroma_db_store", embedding_function=embeddings)

def process_uploaded_file(uploaded_file, vector_store, c_size, c_overlap):
    ext = uploaded_file.name.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    try:
        log_to_ui(f"🔄 Ingesting: {uploaded_file.name}")
        
        if ext == 'pdf': loader = PyPDFLoader(temp_file_path)
        elif ext == 'txt': loader = TextLoader(temp_file_path, encoding='utf-8')
        elif ext == 'docx': loader = Docx2txtLoader(temp_file_path)
        else: return False
            
        raw_docs = loader.load()
        if not raw_docs:
            log_to_ui("⚠️ ERROR: No text found in file.")
            return False
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=c_size, 
            chunk_overlap=c_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        raw_chunks = text_splitter.split_documents(raw_docs)
        
        chunked_docs = []
        for chunk in raw_chunks:
            clean_text = chunk.page_content.replace('═', '').strip()
            if len(clean_text) > 5:
                chunked_docs.append(chunk)
                
        if not chunked_docs:
            log_to_ui("⚠️ ERROR: Zero valid chunks created. Check file content.")
            return False
        
        chunk_data = []
        for i, chunk in enumerate(chunked_docs):
            chunk_data.append({
                "Chunk ID": i,
                "Chunk Size (Chars)": len(chunk.page_content),
                "Content Preview": chunk.page_content[:70].replace('\n', ' ') + "..."
            })
        st.session_state.last_chunk_log = pd.DataFrame(chunk_data)
        
        vector_store.add_documents(chunked_docs)
        log_to_ui(f"✅ Created {len(chunked_docs)} chunks (Size: {c_size}, Overlap: {c_overlap})")
        return True
    except Exception as e:
        log_to_ui(f"❌ ERROR: {str(e)}")
        return False
    finally:
        if os.path.exists(temp_file_path): os.remove(temp_file_path)

# ==========================================
# SIDEBAR UI: CONTROLS & LOGS
# ==========================================
with st.sidebar:
    st.header("⚙️ System Settings")
    api_key = st.text_input("Gemini API Key:", type="password")
    
    chat_selection = "gemini-1.5-flash"
    embed_selection = "text-embedding-004"
    
    if api_key:
        c_models, e_models = get_available_models(api_key)
        
        c_index = 0
        for i, m in enumerate(c_models):
            if "gemini-1.5-flash" in m: c_index = i
            
        e_index = 0
        for i, m in enumerate(e_models):
            if "text-embedding-004" in m: e_index = i
            
        if c_models:
            chat_selection = st.selectbox("Chat Model", c_models, index=c_index)
        if e_models:
            embed_selection = st.selectbox("Embedding Model", e_models, index=e_index)

    st.divider()
    st.subheader("📏 Chunking Strategy")
    st.caption("💡 Sample: Size=1000, Overlap=200 is standard for documentation. Size=500, Overlap=50 is better for short FAQs. Chunk Size --> min_value=100 ~ max_value=5000. Chunk Overlap --> min_value=0 ~max_value=1000")
    c_size = st.number_input("Chunk Size", min_value=100, max_value=5000, value=1000, step=100)
    c_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=1000, value=200, step=50)

    st.divider()
    uploaded_file = st.file_uploader("Upload Knowledge", type=['pdf', 'txt', 'docx'])
    if st.button("Update Knowledge Base"):
        if api_key and uploaded_file:
            db = get_vector_store(api_key, embed_selection)
            process_uploaded_file(uploaded_file, db, c_size, c_overlap)
        else: st.error("Key or File missing.")

    st.divider()
    st.header("📋 Processing Details")
    if "last_chunk_log" in st.session_state:
        st.dataframe(st.session_state.last_chunk_log, hide_index=True)
    
    st.code(st.session_state.log_buffer.getvalue(), language="text")
    if st.button("Clear System Logs"):
        st.session_state.log_buffer = StringIO()
        st.rerun()

# ==========================================
# MAIN CHAT INTERFACE
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "tokens" in message:
            st.caption(f"Tokens: In={message['tokens']['in']} | Out={message['tokens']['out']}")

if prompt := st.chat_input("Ask about your uploaded data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving from ChromaDB..."):
            db = get_vector_store(api_key, embed_selection)
            
            c_model_formatted = chat_selection if chat_selection.startswith("models/") else f"models/{chat_selection}"
            llm = ChatGoogleGenerativeAI(model=c_model_formatted, temperature=0.3)
            
            retriever = db.as_retriever(search_kwargs={"k": 3})
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "Answer strictly using the context below. If missing, say you don't know.\n\nContext:\n{context}"),
                ("human", "{input}"),
            ])
            
            qa_chain = create_stuff_documents_chain(llm, prompt_template)
            rag_chain = create_retrieval_chain(retriever, qa_chain)
            
            response = rag_chain.invoke({"input": prompt})
            
            # 🚨 THE FIX: Native Token Tracking
            # Bypassing LangChain to get the exact counts directly from the Google API
            try:
                native_client = genai.Client(api_key=api_key)
                
                # Combine user prompt and the retrieved context documents
                full_input_context = prompt + "\n" + "\n".join([doc.page_content for doc in response["context"]])
                
                in_tokens = native_client.models.count_tokens(
                    model=c_model_formatted, 
                    contents=full_input_context
                ).total_tokens
                
                out_tokens = native_client.models.count_tokens(
                    model=c_model_formatted, 
                    contents=response["answer"]
                ).total_tokens
                
                token_info = {"in": in_tokens, "out": out_tokens}
            except Exception as e:
                token_info = {"in": "Error", "out": "Error"}
            
            st.markdown(response["answer"])
            st.caption(f"📊 Token usage: Input {token_info['in']} | Output {token_info['out']}")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response["answer"],
                "tokens": token_info
            })
