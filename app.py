import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from src.helper import download_embeddings
from src.prompt import prompt
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

app = Flask(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

index_name = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")

# Initialize Embeddings
embeddings = download_embeddings()

# Load Vector Store & Retriever if API key present
docsearch = None
retriever = None
rag_chain = None

if PINECONE_API_KEY and GOOGLE_API_KEY:
    try:
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        chat_model = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-pro"),
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3
        )

        question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    except Exception as e:
        print(f"[Warning] Failed to initialize Pinecone RAG chain: {e}")

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "")
    if not msg:
        return "Please provide a valid question."

    if rag_chain:
        try:
            response = rag_chain.invoke({"input": msg})
            return response.get("answer", "No answer found in medical context.")
        except Exception as e:
            return f"Error processing query: {str(e)}"
    else:
        # Fallback simulation if running in demo mode without external API keys
        return (
            f"[Demo Mode] Received query: '{msg}'. "
            "To enable live RAG responses, configure PINECONE_API_KEY and GOOGLE_API_KEY in your .env file."
        )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
