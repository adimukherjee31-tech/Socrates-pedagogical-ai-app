import streamlit as st
import os
import tempfile
import time
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

# --- CSS NUCLEAR OPTION: REMOVE BLACK DOTS & FORCE COLOR ---
st.markdown("""
    <style>
    li::marker { content: none !important; }
    ul { list-style-type: none !important; padding-left: 0 !important; }
    li { list-style-type: none !important; padding-left: 0 !important; margin-bottom: 15px !important; }
    .stMarkdown p, .stMarkdown li { 
        font-size: 1.15rem !important;
        line-height: 1.7 !important;
        font-family: "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 Socrates: Pedagogical AI Tutor")

# --- THE QUOTA-SAVER: BATCH EMBEDDING CLASS ---
class QuotaSafeEmbeddings(Embeddings):
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        # Identify available model
        all_models = [m.name for m in genai.list_models() if 'embedContent' in m.supported_generation_methods]
        self.model_name = 'models/text-embedding-004' if 'models/text-embedding-004' in all_models else 'models/embedding-001'

    def embed_documents(self, texts):
        """Sends texts in batches of 100 to avoid Quota 429 errors."""
        all_embeddings = []
        # Google allows up to 100 texts per batch request
        for i in range(0, len(texts), 90):
            batch = texts[i : i + 90]
            try:
                result = genai.embed_content(
                    model=self.model_name,
                    content=batch,
                    task_type="retrieval_document"
                )
                all_embeddings.extend(result["embedding"])
                # Short pause to be extra safe with the free tier rate limit
                time.sleep(1) 
            except Exception as e:
                st.error(f"Batch Error: {e}")
                time.sleep(5) # Wait longer if we hit a snag
        return all_embeddings

    def embed_query(self, text):
        result = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_query"
        )
        return result["embedding"]

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
    
    st.info("✨ **Pinterest Stickers**: Vibrant & Colorful.")
    page_range = st.slider("Select Page Range", 1, 2500, (1, 100))
    start_pg, end_pg = page_range

# --- PROCESSING ---
if api_key and uploaded_file:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.7)

        @st.cache_resource(show_spinner=False)
        def get_vector_db(file_content, _start, _end, _key):
            # Use our new Batch-Enabled fix
            embeddings = QuotaSafeEmbeddings(_key)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name
            
            loader = PyMuPDFLoader(tmp_path)
            all_docs = loader.load()
            docs = all_docs[_start-1 : min(_end, len(all_docs))]
            
            # Increase chunk size slightly to reduce total number of chunks (saves quota)
            splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
            chunks = splitter.split_documents(docs)
            
            db = FAISS.from_documents(chunks, embeddings)
            os.remove(tmp_path)
            return db

        with st.spinner(f"🚀 Indexing section (Batch Mode active to save Quota)..."):
            vector_db = get_vector_db(uploaded_file.getvalue(), start_pg, end_pg, api_key)
            st.sidebar.success(f"✅ Ready! Using {QuotaSafeEmbeddings(api_key).model_name}")

        # --- CHAT ---
        query = st.chat_input("Ask a question from the book...")
        
        if query:
            with st.chat_message("user"): st.write(query)

            context_docs = vector_db.similarity_search(query, k=5)
            context_text = "\n\n".join([d.page_content for d in context_docs])

            styles = {
                "Professor": "Academic Tutor. Professional yet aesthetic.",
                "Munnabhai (Hinglish)": "Munnabhai style. Use Hinglish, 'Mammu', and funny life analogies.",
                "Physicswallah UGC-NET Coach": "High-energy coach. 'Hello Baccho!', 'Ekdum basic se samjhenge', 'Selection rukna nahi chahiye!'. Use Hinglish.",
                "Simple": "Explain like I'm 10 with bright, colorful examples."
            }

            prompt = ChatPromptTemplate.from_template("""
            You are Socrates, a pedagogical tutor. 
            
            GROUNDING:
            - If found in Context: Answer and MUST end with "[SOURCE: TEXTBOOK]"
            - If not: Answer and MUST start with "[SOURCE: GENERAL AI KNOWLEDGE]"

            PINTEREST STICKER RULES (STRICT):
            - NEVER use black dots, gray circles, or dashes (-, *, •).
            - START EVERY POINT with a unique, BRIGHT, COLORFUL Pinterest emoji.
            - USE THESE: 🌈, 🍭, 🎀, ✨, 🎨, 🌟, 🍬, 🦋, 🦄, 🎈, 🧁, 🌸, 🎡, 🍓, 🍦, 🍭, 🎡, 🍄.
            - SUB-POINTS: Use "╰┈➤ 💖" and a different colorful emoji.
            - The final output must look like a VIBRANT digital scrapbook.
            
            Context: {context}
            Style/Personality: {personality}
            Question: {question}
            
            Answer:""")

            chain = prompt | llm | StrOutputParser()

            with st.chat_message("assistant"):
                response = chain.invoke({"personality": styles[tone], "context": context_text, "question": query})
                st.markdown(response)

    except Exception as e:
        if "429" in str(e):
            st.error("Too many requests! Google is cooling down. Please wait 1 minute and try again.")
        else:
            st.error(f"System Error: {e}")
else:
    st.warning("Please enter your API Key and upload a PDF.")
