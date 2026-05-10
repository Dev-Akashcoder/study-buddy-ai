import streamlit as st
import os
import tempfile
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_option_menu import option_menu
import requests
from streamlit_lottie import st_lottie

# --- LANGCHAIN IMPORTS ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. PAGE CONFIGURATION & CODEX UI CSS ---
st.set_page_config(page_title="Study Buddy AI", page_icon="🎓", layout="wide")

# Helper function to load animations from URL
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Custom CSS for the Codex Aesthetic
st.markdown("""
    <style>
    /* 1. The Bright Blue/Purple Aura Background */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 20% 30%, #b5e8ff 0%, #4a8cff 45%, #7a3bf5 100%);
        background-attachment: fixed;
    }
    
    /* 2. Glassmorphism Sidebar (Light) */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* 3. Make Main Text Dark & Centered */
    h1, h2, h3 {
        color: #111111 !important;
        text-align: center !important;
        font-weight: 600 !important;
    }
    
    /* 4. The "Download for Windows" Black Pill Button */
    .stButton>button {
        border-radius: 50px !important;
        background: #111111 !important;
        color: white !important;
        border: none !important;
        padding: 10px 24px !important;
        font-weight: 500 !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
        transition: all 0.2s ease;
        display: block;
        margin: 0 auto; /* Centers the button */
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background: #333333 !important;
        color: white !important;
    }

    /* 5. The Dark "Terminal/Editor" Window at the bottom (Chat Bubbles) */
    .stChatMessage {
        background: rgba(20, 20, 22, 0.95) !important;
        color: #ffffff !important;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        margin-top: 15px;
    }
    .stChatMessage * {
        color: #e2e8f0 !important; /* Make chat text light gray/white */
    }
    
    /* Hide default Streamlit fluff */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] {background: transparent !important;}
    </style>
""", unsafe_allow_html=True)

# Load Lottie Animations
lottie_login = load_lottieurl("https://lottie.host/80517f8a-c63d-4654-be8e-f55e5330a133/M8wXz0mUeQ.json") 
lottie_upload = load_lottieurl("https://lottie.host/a61c3132-8df2-463d-9d41-e946a36f6d52/K8gK7oD6t3.json") 

# --- 2. AUTHENTICATION (WITH REGISTRATION) ---
# Load the user database
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Initialize Authenticator (Updated for newer Streamlit-Authenticator version)
if "authenticator" not in st.session_state:
    st.session_state.authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

# Login Page UI
if st.session_state.get("authentication_status") != True:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if lottie_login:
            st_lottie(lottie_login, height=250, key="login_anim")
        st.session_state.authenticator.login()

# Handle Login/Register Logic
if st.session_state.get("authentication_status") is False:
    st.error("❌ Username/password is incorrect")

elif st.session_state.get("authentication_status") is None:
    # Show Registration Form below Login
    st.markdown("---")
    st.write("### Don't have an account?")
    try:
        # Open registration for anyone (no pre_authorized argument)
        email_of_registered_user, username_of_registered_user, name_of_registered_user = st.session_state.authenticator.register_user()
        if email_of_registered_user:
            st.success('User registered successfully! You can now log in above.')
            # Save the new user permanently to config.yaml
            with open('config.yaml', 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
    except Exception as e:
        st.error(e)

elif st.session_state.get("authentication_status"):
    
    # --- 3. PREMIUM NAVIGATION MENU ---
    with st.sidebar:
        st.write(f"Welcome back, **{st.session_state['name']}**!")
        selected = option_menu(
            menu_title="Main Menu",
            options=["AI Chat", "Upload Documents", "Settings"],
            icons=["chat-dots", "cloud-upload", "gear"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#111", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "rgba(255,255,255,0.2)", "border-radius":"10px", "color": "#111"},
                "nav-link-selected": {"background-color": "rgba(255,255,255,0.4)", "box-shadow": "0 4px 10px rgba(0,0,0,0.1)", "color": "#111", "font-weight": "bold"},
            }
        )
        st.divider()
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.toast("Chat history cleared!", icon="✅")
        st.session_state.authenticator.logout("Logout", "sidebar")

    # Initialize Chat History Memory
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --- 4. APP LOGIC ---
    if selected == "AI Chat":
        # Codex Style Centered Title Header
        st.markdown("""
            <div style='text-align: center; margin-bottom: 30px;'>
                <h1 style='font-size: 3.5rem; margin-bottom: 0px;'>Study Buddy AI</h1>
                <p style='color: #222; font-size: 1.2rem;'>A smart agent that helps you learn and study with AI.</p>
            </div>
        """, unsafe_allow_html=True)
        
        if "vectorstore" not in st.session_state:
            st.info("👈 Please go to 'Upload Documents' in the sidebar to upload your course materials first!")
        else:
            # Setup AI components
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=st.secrets["GOOGLE_API_KEY"])
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
            
            system_prompt = (
                "You are an intelligent university study assistant. "
                "Use the following pieces of retrieved context to answer the student's question. "
                "If the answer is not in the context, explicitly say 'I do not know the answer based on the uploaded documents.' "
                "Do not use outside knowledge. \n\nContext: {context}"
            )
            prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
            
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            # Display previous chat history (Codex dark mode style)
            for message in st.session_state.chat_history:
                avatar_icon = "🧑‍🎓" if message["role"] == "user" else "🤖"
                with st.chat_message(message["role"], avatar=avatar_icon):
                    st.write(message["content"])
                    if "sources" in message and message["sources"]:
                        with st.expander("📄 View Sources"):
                            for i, doc in enumerate(message["sources"]):
                                st.caption(f"**Source {i+1}:** {doc.page_content[:300]}...")

            # Chat Input Form
            user_query = st.chat_input("Ask a question about your documents...")
            if user_query:
                with st.chat_message("user", avatar="🧑‍🎓"):
                    st.write(user_query)
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Analyzing documents..."):
                        response = rag_chain.invoke({"input": user_query})
                        answer = response["answer"]
                        sources = response.get("context", [])
                        
                        st.write(answer)
                        
                        if sources:
                            with st.expander("📄 View Sources"):
                                for i, doc in enumerate(sources):
                                    st.caption(f"**Source {i+1}:** {doc.page_content[:300]}...")
                                    
                st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})

    elif selected == "Upload Documents":
        col1, col2 = st.columns([2, 1])
        with col1:
            st.title("📂 Document Library")
            st.write("Upload multiple PDFs (Syllabus, Notes, Textbooks) to build your AI's knowledge base.")
            
            uploaded_files = st.file_uploader("Upload course materials", type="pdf", accept_multiple_files=True)
            
            if st.button("Process Documents") and uploaded_files:
                with st.spinner("Extracting and processing text from all documents..."):
                    all_splits = []
                    
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                            temp_file.write(uploaded_file.getvalue())
                            temp_file_path = temp_file.name
                        
                        loader = PyPDFLoader(temp_file_path)
                        docs = loader.load()
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        splits = text_splitter.split_documents(docs)
                        all_splits.extend(splits)
                    
                    if not all_splits:
                        st.error("❌ Error: I couldn't find any readable text in these PDFs.")
                    else:
                        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                        st.session_state.vectorstore = FAISS.from_documents(documents=all_splits, embedding=embeddings)
                        st.toast("All documents processed successfully! Head to the AI Chat.", icon="✅")
                        
        with col2:
            if lottie_upload:
                st_lottie(lottie_upload, height=300, key="upload_anim")
                    
    elif selected == "Settings":
        st.title("⚙️ Settings")
        st.write("Current LLM Engine: **Google Gemini 2.5 Flash**")
        st.write("Vector Database: **FAISS (Local Memory)**")
        st.write("Chunk Size: **1000** | Overlap: **200**")
        st.info("You can clear your chat history using the button in the sidebar.")