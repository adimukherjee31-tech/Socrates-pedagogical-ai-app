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

# CSS: FORCE COLORFUL EMOJIS & REMOVE BLACK BULLETS
st.markdown("""
    <style>
    /* 1. Remove all standard black dots/bullets */
    ul, li { list-style-type: none !important; padding-left: 0 !important; margin-left: 0 !important; }
    
    /* 2. Force emojis to be colorful and font-friendly */
    .stMarkdown p, .stMarkdown li { 
        font-family: "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif !important;
        font-size: 1.1rem !important;
        line-height: 1.8 !important;
    }
    
    /* 3. Make chat headers pretty */
    h3 { color: #FF69B4; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 Socrates: Pedagogical AI Tutor")

# --- THE "NO-MORE-404" EMBEDDING CLASS ---
class UniversalGoogleEmbeddings(Embeddings):
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        # We try the newest model first, then fallback to the legacy name
        self.model_name = "models/text-embedding-004"
        try:
            genai.embed_content(model=self.model_name, content="test")
        except:
            self.model_name = "models/embedding-001"

    def embed_documents(self, texts):
        # Direct API call bypasses the LangChain 'v1beta' bug
        return [genai.embed_content(model=self.model_name, content=t, task_type="retrieval_document")["embedding"] for t in texts]

    def embed_query(self, text):
        return genai.embed_content(model=self.model_name, content=text, task_type="retrieval_query")["embedding"]

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
    
    st.info("✨ **Pinterest Stickers**: High-Vibrancy Emojis Only.")
    page_range = st.slider("Select Page Range", 1, 2500, (1, 100))
    start_pg, end_pg = page_range

# --- PROCESSING ---
if api_key and uploaded_file:
    try:
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.7)
        
        @st.cache_resource(show_spinner=False)
        def get_vector_db(file_content, _start, _end, _key):
            # Use our direct Embedding fix
            embeddings = UniversalGoogleEmbeddings(_key)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name
            
            loader = PyMuPDFLoader(tmp_path)
            all_docs = loader.load()
            docs = all_docs[_start-1 : min(_end, len(all_docs))]
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(docs)
            
            # Create the vector store
            db = FAISS.from_documents(chunks, embeddings)
            os.remove(tmp_path)
            return db

        with st.spinner(f"🚀 Indexing pages {start_pg} to {end_pg}..."):
            vector_db = get_vector_db(uploaded_file.getvalue(), start_pg, end_pg, api_key)
            st.sidebar.success("✅ Book Indexed Successfully!")

        # --- CHAT ---
        query = st.chat_input("Ask a question...")
        
        if query:
            with st.chat_message("user"): st.write(query)

            # Search for relevant context
            context_docs = vector_db.similarity_search(query, k=5)
            context_text = "\n\n".join([d.page_content for d in context_docs])

            styles = {
                "Professor": "Academic Tutor. Professional but uses aesthetic markers.",
                "Munnabhai (Hinglish)": "Munnabhai style. Use Hinglish, call user 'Mammu', use funny analogies.",
                "Physicswallah UGC-NET Coach": "High-energy, motivational coaching style. Use 'Hello Baccho!', 'Selection rukna nahi chahiye!'. Use Hinglish.",
                "Simple": "Explain like I'm 10 with colorful examples."
            }

            prompt = ChatPromptTemplate.from_template("""
            You are Socrates, a pedagogical tutor. Use the Context to answer the Question.
            
            GROUNDING:
            - If found in Context: Answer and end with "[SOURCE: TEXTBOOK]"
            - If not: Answer and start with "[SOURCE: GENERAL AI KNOWLEDGE]"

            PINTEREST STICKER RULES (MANDATORY):
            - NO BLACK DOTS. NO ASTERISKS. NO DASHES.
            - Start EVERY single point with a bright, colorful Pinterest emoji sticker.
            - Use a variety of: 🌈, 🍭, 🎀, ✨, 🎨, 🌟, 🍬, 🦋, 🦄, 🎈, 🧁, 🌸, 🎡, 🍓, 🍦, 🍭.
            - Sub-points must start with "╰┈➤ 💖" and another colorful emoji.
            - The final output must look like a VIBRANT digital study scrapbook.
            
            Context: {context}
            Style/Personality: {personality}
            Question: {question}
            
            Answer:""")

            chain = prompt | llm | StrOutputParser()

            with st.chat_message("assistant"):
                response = chain.invoke({"personality": styles[tone], "context": context_text, "question": query})
                st.markdown(response)

    except Exception as e:
        st.error(f"System Error: {e}")
else:
    st.warning("Please enter your API Key and upload a PDF.")
