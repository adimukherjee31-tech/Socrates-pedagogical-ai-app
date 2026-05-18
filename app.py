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

# --- CSS NUCLEAR OPTION: REMOVE BLACK DOTS & FORCE COLOR ---
st.markdown("""
    <style>
    /* Force delete all standard browser bullets */
    li::marker { content: none !important; }
    ul { list-style-type: none !important; padding-left: 0 !important; }
    li { list-style-type: none !important; padding-left: 0 !important; margin-bottom: 15px !important; }
    
    /* Make text readable and emoji-friendly */
    .stMarkdown p, .stMarkdown li { 
        font-size: 1.15rem !important;
        line-height: 1.7 !important;
        font-family: "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 Socrates: Pedagogical AI Tutor")

# --- THE DEFINITIVE FIX: DYNAMIC MODEL DISCOVERY EMBEDDINGS ---
class DynamicGoogleEmbeddings(Embeddings):
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model_name = None
        
        # Ask Google: "Which models do I have access to?"
        available_models = [m.name for m in genai.list_models() if 'embedContent' in m.supported_generation_methods]
        
        # Selection Priority: text-embedding-004 > embedding-001 > anything else
        if 'models/text-embedding-004' in available_models:
            self.model_name = 'models/text-embedding-004'
        elif 'models/embedding-001' in available_models:
            self.model_name = 'models/embedding-001'
        elif available_models:
            self.model_name = available_models[0]
        else:
            raise Exception("No embedding models found for this API Key. Please check your Google AI Console.")

    def embed_documents(self, texts):
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
    
    st.info("✨ **Pinterest Stickers**: Vibrant & Colorful Mode.")
    page_range = st.slider("Select Page Range", 1, 2500, (1, 100))
    start_pg, end_pg = page_range

# --- PROCESSING ---
if api_key and uploaded_file:
    try:
        # Initialize the Chat Model
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.7)

        @st.cache_resource(show_spinner=False)
        def get_vector_db(file_content, _start, _end, _key):
            # Use our Dynamic Discovery fix
            embeddings = DynamicGoogleEmbeddings(_key)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name
            
            loader = PyMuPDFLoader(tmp_path)
            all_docs = loader.load()
            docs = all_docs[_start-1 : min(_end, len(all_docs))]
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(docs)
            
            db = FAISS.from_documents(chunks, embeddings)
            os.remove(tmp_path)
            return db

        with st.spinner(f"🚀 Scanning library..."):
            vector_db = get_vector_db(uploaded_file.getvalue(), start_pg, end_pg, api_key)
            # Show which model was actually discovered/used
            discovered_model = DynamicGoogleEmbeddings(api_key).model_name
            st.sidebar.success(f"✅ Ready! (Using: {discovered_model.split('/')[-1]})")

        # --- CHAT ---
        query = st.chat_input("Ask a question from the book...")
        
        if query:
            with st.chat_message("user"): st.write(query)

            context_docs = vector_db.similarity_search(query, k=5)
            context_text = "\n\n".join([d.page_content for d in context_docs])

            styles = {
                "Professor": "Academic Tutor. Professional yet aesthetic.",
                "Munnabhai (Hinglish)": "Munnabhai style. Use Hinglish, call user 'Mammu', use funny analogies.",
                "Physicswallah UGC-NET Coach": "High-energy, motivational coaching style. Use 'Hello Baccho!', 'Ekdum basic se samjhenge', 'Selection rukna nahi chahiye!'. Use Hinglish.",
                "Simple": "Explain like I'm 10 with colorful examples."
            }

            prompt = ChatPromptTemplate.from_template("""
            You are Socrates, a pedagogical tutor. Use the Context to answer the Question.
            
            GROUNDING:
            - If found in Context: Answer and MUST end with "[SOURCE: TEXTBOOK]"
            - If not: Answer and MUST start with "[SOURCE: GENERAL AI KNOWLEDGE]"

            PINTEREST STICKER RULES (STRICT):
            - NEVER use black dots, gray circles, asterisks, or dashes (-, *, •, 🔘).
            - START EVERY POINT with a unique, BRIGHT, COLORFUL emoji sticker.
            - USE ONLY VIVID EMOJIS: 🌈, 🍭, 🎀, ✨, 🎨, 🌟, 🍬, 🦋, 🦄, 🎈, 🧁, 🌸, 🎡, 🍓, 🍦, 🍭, 🎠, 🎨.
            - SUB-POINTS: Use "╰┈➤ 💖" followed by a DIFFERENT colorful emoji.
            - The final output must look like a VIBRANT, colorful digital scrapbook.
            
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
