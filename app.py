import streamlit as st
import os
import tempfile
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- PAGE CONFIG ---
st.set_page_config(page_title="Socrates AI Tutor", layout="wide", page_icon="🎓")

# Custom CSS to make sure list items don't show default black bullets
st.markdown("""
    <style>
    ul { list-style-type: none; }
    li { list-style-type: none; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 Socrates: Pedagogical AI Tutor")

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
    
    st.info("🌈 **Color Mode Active**: Using high-vibrancy aesthetic markers.")
    page_range = st.slider(
        "Select Page Range to Index", 
        1, 2500, (1, 200) 
    )
    start_page, end_page = page_range

# --- AUTO-DETECT MODELS ---
def get_working_model(api_key):
    try:
        genai.configure(api_key=api_key)
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        clean_models = [m.replace('models/', '') for m in models]
        for preferred in ['gemini-1.5-flash', 'gemini-1.5-pro']:
            if preferred in clean_models:
                return preferred
        return clean_models[0] if clean_models else None
    except Exception:
        return "gemini-1.5-flash" 

# --- PROCESSING ---
if api_key and uploaded_file:
    try:
        active_model = get_working_model(api_key)
        st.sidebar.success(f"Connected to: {active_model}")

        llm = ChatGoogleGenerativeAI(
            model=active_model,
            google_api_key=api_key,
            temperature=0.6 # Increased for more colorful and diverse emoji selection
        )

        @st.cache_resource(show_spinner=False)
        def get_vector_db(file_content, start_pg, end_pg, _api_key):
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", 
                google_api_key=_api_key
            )
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name
            
            loader = PyMuPDFLoader(tmp_path)
            all_docs = loader.load()
            end_pg = min(end_pg, len(all_docs))
            docs = all_docs[start_pg-1 : end_pg]
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            db = FAISS.from_documents(chunks, embeddings)
            os.remove(tmp_path)
            return db

        with st.spinner(f"🚀 Speed-indexing pages {start_page} to {end_page}..."):
            vector_db = get_vector_db(uploaded_file.getvalue(), start_page, end_page, api_key)

        # --- CHAT ---
        query = st.chat_input("Ask a question from this section...")
        
        if query:
            with st.chat_message("user"):
                st.write(query)

            context_docs = vector_db.similarity_search(query, k=5)
            context_text = "\n\n".join([d.page_content for d in context_docs])

            styles = {
                "Professor": "Professional Academic Tutor. Use bright, colorful aesthetic headers.",
                "Munnabhai (Hinglish)": "Munnabhai style. Use Hinglish, call user 'Mammu', use funny analogies.",
                "Physicswallah UGC-NET Coach": "High-energy coach. Use 'Hello Baccho!', 'Selection rukna nahi chahiye!'. Focus on exam patterns.",
                "Simple": "Explain like I'm 10 years old with colorful analogies."
            }

            prompt = ChatPromptTemplate.from_template("""
            You are Socrates, a pedagogical tutor. Use the provided Context to answer the Question.
            
            GROUNDING RULES:
            1. Use Context first. Append "[SOURCE: TEXTBOOK]" if found.
            2. If not in Context, use General Knowledge and start with "[SOURCE: GENERAL AI KNOWLEDGE]".
            
            COLORFUL AESTHETIC RULES (STRICT):
            - **CRITICAL**: Use ONLY high-vibrancy, bright-colored emojis.
            - **FORBIDDEN**: Do not use standard bullets (- or * or •). Do not use black and white emojis or monochrome symbols.
            - **VIBRANT POINTERS**: Start every point with a DIFFERENT bright emoji like 🌈, 🧚‍♀️, 🎀, 🍭, ✨, 🎨, 🌟, 🍬, 🍄, 🦄, 🎡, 🍉, 🧁, 🦋, 🎈.
            - **SUB-POINTS**: Use ╰┈➤ followed by a colorful emoji.
            - **STYLE**: The response should look like a bright, Pinterest-inspired digital scrapbook. Every section must pop with color.
            
            Context: {context}
            Style/Personality: {personality}
            Question: {question}
            
            Answer:""")

            chain = prompt | llm | StrOutputParser()

            with st.chat_message("assistant"):
                try:
                    response = chain.invoke({
                        "personality": styles[tone],
                        "context": context_text,
                        "question": query
                    })
                    st.markdown(response)
                except Exception as e:
                    st.error(f"AI Connection Error: {e}")

    except Exception as general_err:
        st.error(f"System Error: {general_err}")
else:
    st.warning("Enter API Key and upload PDF to start.")
