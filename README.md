# 🩺 MediBot — End-to-End Medical Chatbot with LangChain, Pinecone & Google Gemini

<p align="center">
  <a href="https://medical-chatbot-bk77.onrender.com"><img src="https://img.shields.io/badge/Live_Demo-Render-46E3B7?logo=render&logoColor=white" alt="Live Demo" /></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/LangChain-0.3.x-1C3C3C?logo=langchain" alt="LangChain" />
  <img src="https://img.shields.io/badge/Pinecone-Serverless%20Vector%20DB-000000?logo=pinecone" alt="Pinecone" />
  <img src="https://img.shields.io/badge/Google%20Gemini-LLM-4285F4?logo=google" alt="Google Gemini" />
  <img src="https://img.shields.io/badge/Flask-Web%20Framework-000000?logo=flask" alt="Flask" />
  <img src="https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-yellow" alt="Sentence Transformers" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License" />
</p>

An enterprise-grade **Retrieval-Augmented Generation (RAG) conversational medical assistant** designed to parse comprehensive clinical literature, index vector embeddings in a serverless vector database, and generate clinically grounded, empathetic responses using Google's Gemini LLMs.

> 🌐 **Live Demo:** Try the deployed chatbot here: [https://medical-chatbot-bk77.onrender.com](https://medical-chatbot-bk77.onrender.com)

---

## 🏗️ RAG System Architecture

`	ext
       ┌────────────────────────┐
       │   Medical Literature   │
       │       (PDF Data)       │
       └───────────┬────────────┘
                   │ PyPDFLoader + RecursiveCharacterTextSplitter (500 chars, 20 overlap)
                   ▼
       ┌────────────────────────┐
       │     Document Chunks    │
       └───────────┬────────────┘
                   │ HuggingFace (sentence-transformers/all-MiniLM-L6-v2)
                   ▼
       ┌────────────────────────┐
       │   384-dim Embeddings   │
       └───────────┬────────────┘
                   │ Upsert Index
                   ▼
       ┌────────────────────────┐
       │  Pinecone Vector DB    │ ◄── Cosine Similarity Retrieval (k=3)
       └───────────┬────────────┘                       ▲
                   │                                    │ Query Vector
                   ▼ Retrieved Context                  │
       ┌────────────────────────────────────────────────┴───────┐
       │                    User Query Interface                │
       │                   (Flask + Responsive UI)              │
       └───────────────────────────┬────────────────────────────┘
                                   │ Context + Strict Medical Prompt Template
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │                 Google Gemini Pro LLM                  │
       │        (Grounded Clinical Answer Generation)           │
       └───────────────────────────┬────────────────────────────┘
                                   ▼
                            Verified Response
`

---

## 📁 Project Structure

`	ext
medical-chatbot/
├── data/
│   └── Medical_book.pdf            # Comprehensive clinical reference literature
├── research/
│   └── trials.ipynb                # Prototyping notebook for chunking & retrieval
├── src/
│   ├── __init__.py
│   ├── helper.py                   # PDF extraction, text splitting & embedding loaders
│   └── prompt.py                   # Strict clinical prompt template for hallucination reduction
├── templates/
│   └── chat.html                   # Responsive web application interface
├── app.py                          # Flask web server & RAG query execution pipeline
├── store_index.py                  # Batch embedding generation & Pinecone vector upsert script
├── requirements.txt                # Production dependencies
├── setup.py                        # Package distribution metadata
├── .env.example                    # Environment variable configuration template
└── README.md
`

---

## ✨ Key Capabilities & Engineering Highlights

- **Retrieval-Augmented Generation (RAG)**: Combines semantic search across curated medical textbooks with LLM generation to eliminate hallucinations and anchor responses strictly in verified literature.
- **Serverless Vector Indexing**: Utilizes **Pinecone Serverless** (dimension=384, metric="cosine") for sub-100ms vector similarity matching.
- **Dense Vector Representations**: Employs **HuggingFace sentence-transformers/all-MiniLM-L6-v2** embeddings for lightweight, high-accuracy semantic retrieval.
- **Controlled Hallucination Mitigation**: Uses customized system instructions requiring the model to explicitly state uncertainty when clinical evidence is absent from the knowledge base.
- **Full-Stack Interactive Interface**: Powered by a lightweight Flask backend paired with an asynchronous web chat interface for real-time interaction.

---

## 🌐 Live Demo

Experience the live deployed medical chatbot on Render:

🔗 **[https://medical-chatbot-bk77.onrender.com](https://medical-chatbot-bk77.onrender.com)**

> *Note: Hosted on Render free tier. If the instance has been idle, please allow 30–50 seconds for the initial cold start.*

---

## 🚀 Getting Started

### 1. Clone and Setup Environment

`ash
git clone https://github.com/saigoutham17/medical-chatbot.git
cd medical-chatbot

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
`

### 2. Configure Environment Variables

Create a .env file based on .env.example:

`env
PINECONE_API_KEY="your_pinecone_api_key"
PINECONE_INDEX_NAME="medical-chatbot"
GOOGLE_API_KEY="your_google_gemini_api_key"
GEMINI_MODEL="gemini-pro"
PORT=8080
`

### 3. Ingest Documents & Build Vector Store

Execute the ingestion pipeline to parse the medical reference books, generate vector embeddings, and upsert them into your Pinecone index:

`ash
python store_index.py
`

### 4. Launch the Medical Chatbot

`ash
python app.py
`

Navigate to http://localhost:8080 in your browser to interact with the medical assistant.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | / | Renders interactive chat web interface |
| POST | /get | Processes user query against Pinecone index and returns Gemini RAG response |

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

**Sai Goutham**
* GitHub: [@saigoutham17](https://github.com/saigoutham17)
* Project: [Medical Chatbot RAG System](https://github.com/saigoutham17/medical-chatbot)
* Live Demo: [medical-chatbot-bk77.onrender.com](https://medical-chatbot-bk77.onrender.com)
