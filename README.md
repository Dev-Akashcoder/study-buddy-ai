# Study Buddy AI 
**A Retrieval-Augmented Educational Assistant**

## Overview
Study Buddy AI is a context-aware digital learning assistant designed to help university students instantly retrieve and synthesize information from lengthy academic PDFs. By utilizing a localized Retrieval-Augmented Generation (RAG) pipeline, this application eliminates AI hallucination by forcing the underlying Large Language Model to answer strictly based on the user's uploaded syllabus or textbooks.

## Key Features
* **Zero Hallucination:** Deterministic AI responses bounded to verified academic documents.
* **High-Speed Vector Search:** Powered by local FAISS indexing and HuggingFace `all-MiniLM-L6-v2` embeddings.
* **Data Privacy:** Cryptographically secured sessions using `bcrypt` password hashing.
* **Distraction-Free UI:** Built entirely in Python using the Streamlit framework.

## Tech Stack
* **Frontend:** Streamlit
* **Orchestration:** LangChain
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Generative Engine:** Google Gemini 2.5 Flash API
* **Language:** Python 3.9+

## Run Locally
1. Clone the repository: `git clone https://github.com/Dev-Akashcoder/StudyBuddyAI.git`
2. Create a virtual environment: `python -m venv .venv`
3. Activate the environment and install dependencies: `pip install -r requirements.txt`
4. Add your Google Gemini API key to a `.env` file.
5. Run the application: `streamlit run app.py`

## Academic Context
Developed as a Major Project Dissertation for the partial fulfillment of the B.Tech degree at Delhi Technical Campus (Affiliated to GGSIPU).
