import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")

def ingest_data():
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set in environment variables.")

    print("📖 Loading PDF documents from data/ directory...")
    extracted_data = load_pdf_files("data")
    print(f"Loaded {len(extracted_data)} document pages.")

    print("🧹 Filtering to minimal document structure...")
    minimal_docs = filter_to_minimal_docs(extracted_data)

    print("✂️ Chunking text into chunks of 500 characters...")
    text_chunks = text_split(minimal_docs, chunk_size=500, chunk_overlap=20)
    print(f"Created {len(text_chunks)} text chunks.")

    print("📥 Loading sentence-transformers embedding model...")
    embeddings = download_embeddings()

    print(f"🌲 Initializing Pinecone client and index '{index_name}'...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if not pc.has_index(index_name):
        print(f"Creating Pinecone index '{index_name}' with dimension 384...")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    print("⚡ Upserting vector embeddings to Pinecone...")
    PineconeVectorStore.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        index_name=index_name
    )
    print("✅ Successfully indexed medical knowledge base in Pinecone!")

if __name__ == "__main__":
    ingest_data()
