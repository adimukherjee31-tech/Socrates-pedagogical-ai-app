import streamlit as st
import os
import tempfile
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings

# --- PAGE CONFIG ---
st.set_page_config(page_title="Socrates AI Tutor", layout="wide", page_icon="🎓")

# CSS HACK: Force colorful emojis and hide standard black dots
st.markdown("""
    <style>
    /* Force Emojis to be colorful and hide default dots */
    ul, li { list-style-type: none !important; }
    .stMarkdown p, .stMarkdown li { 
        font-family: "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji", sans-serif !important;
        font-size: 1.1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 Socrates: Pedagogical AI Tutor")

# --- CUSTOM EMBEDDINGS CLASS (Fixes the 404 Error) ---
class SimpleGoogleEmbeddings(Embeddings):
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
    def embed_documents(self, texts):
        # Uses the most stable embedding method directly
        return [genai.embed_content(model="models/embedding-001", content=t, task_type="retrieval_document")["embedding"] for t in texts]
    def embed_query(self, text):
        return genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_query")["embedding"]

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Setup")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    uploaded_file = st.file_uploader("Upload Textbook/Notes (PDF)", type="pdf")
    
    st.header("2. Study Settings")
    tone = st.selectbox("Teaching Style", [
        "Professor", 
        "Munnabhai (Hinglish)", 
        "Physicswallah UGC-NET Coach", 
        "Simple"
    ])
    
    st.info("🌈 **Aesthetic Mode**: Vibrant Stickers Enabled.")
    page_range = st.slider("Select Page Range", 1, 2500, (1, 100))
    start_page, end_page = page_range

# --- PROCESSING ---
if api_key and uploaded_file:
    try:
        genai.configure(api_key=api_key)
        # Use Gemini 1.5 Flash for Chat
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.7)
        st.sidebar.success("Connected to Gemini 1.5 Flash")

        @st.cache_resource(show_spinner=False)
        def get_vector_db(file_content, start_pg, end_pg, _api_key):
            # Custom Direct Embedding Call
            embeddings = SimpleGoogleEmbeddings(_api_key)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name
            
            loader = PyMuPDFLoader(tmp_path)
            all_docs = loader.load()
            docs = all_docs[start_pg-1 : min(end_pg, len(all_docs))]
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            db = FAISS.from_documents(chunks, embeddings)
            os.remove(tmp_path)
            return db

        with st.spinner(f"🚀 Indexing pages {start_page} to {end_page}..."):
            vector_db = get_vector_db(uploaded_file.getvalue(), start_page, end_page, api_key)

        # --- CHAT ---
        query = st.chat_input("Ask anything from this section...")
        
        if query:
            with st.chat_message("user"): st.write(query)

            context_docs = vector_db.similarity_search(query, k=5)
            context_text = "\n\n".join([d.page_content for d in context_docs])

            styles = {
                "Professor": "Academic Tutor. Professional yet aesthetic.",
                "Munnabhai (Hinglish)": "Munnabhai style. Use Hinglish and 'Mammu'.",
                "Physicswallah UGC-NET Coach": "High-energy coach. 'Hello Baccho!', 'Selection rukna nahi chahiye!'.",
                "Simple": "Explain like I'm 10 with colorful examples."
            }

            prompt = ChatPromptTemplate.from_template("""
            You are Socrates, a pedagogical tutor. 
            
            GROUNDING:
            - If found in Context: Explain and end with "[SOURCE: TEXTBOOK]"
            - If not: Use general knowledge and start with "[SOURCE: GENERAL AI KNOWLEDGE]"

            PINTEREST STICKER RULES (CRITICAL):
            - NEVER use black dots, dashes, or asterisks (•, -, *) for lists.
            - Start EVERY new point with a unique, BRIGHT, COLORFUL emoji sticker.
            - Use ONLY vivid emojis: 🌈, 🍭, 🎀, ✨, 🎨, 🌟, 🍬, 🦋, 🦄, 🎈, 🧁, 🌸, 🎡, 🍓, 🍦.
            - Use "╰┈➤ 💖" for sub-points.
            - The entire output should look like a VIBRANT digital scrapbook.
            
            Context: {context}
            Style: {personality}
            Question: {question}
            
            Answer:""")

            chain = prompt | llm | StrOutputParser()

            with st.chat_message("assistant"):
                response = chain.invoke({"personality": styles[tone], "context": context_text, "question": query})
                st.markdown(response)

    except Exception as e:
        st.error(f"System Error: {e}")
else:
    st.warning("Enter API Key and upload PDF to start.")
