import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. SETUP API KEY ---
# ⚠️ REPLACE THE TEXT BELOW WITH YOUR ACTUAL GOOGLE API KEY! Keep the quotation marks!
os.environ["GOOGLE_API_KEY"] = "AIzaSyAhjnnh6e-V9Kb8iOV4o-Vx4tvrsydaUW0"

# --- 2. STREAMLIT UI & MEMORY ---
st.set_page_config(page_title="Study Buddy", page_icon="📚")
st.title("📚 Personalized Study Buddy (RAG)")

# Create a "memory" for Streamlit so it doesn't forget the PDF every time you type a question
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

# --- 3. DATA INGESTION (Only runs once when a file is uploaded) ---
if uploaded_file is not None and st.session_state.vectorstore is None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    with st.spinner("Reading and memorizing your document... Please wait."):
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Save the database into Streamlit's memory
        st.session_state.vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        st.success("Document processed successfully! You can now ask questions.")

# --- 4. CHAT INTERFACE ---
if st.session_state.vectorstore is not None:
    user_query = st.text_input("Ask a question based on the uploaded PDF:")

    if user_query:
        # Fetch the database from memory
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

        system_prompt = (
            "You are an intelligent university study assistant. "
            "Use the following pieces of retrieved context to answer the student's question. "
            "If the answer is not in the context, explicitly say 'I do not know the answer based on the uploaded document.' "
            "Do not use outside knowledge. "
            "\n\nContext: {context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        with st.spinner("Thinking..."):
            try:
                # Generate the answer
                response = rag_chain.invoke({"input": user_query})
                st.markdown("### Answer:")
                st.write(response["answer"])
                
                with st.expander("Show source context (Where did I find this?)"):
                    for i, doc in enumerate(response["context"]):
                        st.write(f"**Chunk {i+1}:** {doc.page_content}")
                        
            except Exception as e:
                # If it fails, this will print the EXACT error on your web screen nicely
                st.error(f"An error occurred with the AI. Did you paste your API key correctly? \n\nError details: {e}")