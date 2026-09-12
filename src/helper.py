from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

def load_pdf_files(data_path: str):
    """
    Load PDF documents from the specified directory.
    """
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list containing
    only 'source' in metadata and the clean page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source", "")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

def text_split(extracted_docs: List[Document], chunk_size: int = 500, chunk_overlap: int = 20):
    """
    Split documents into smaller chunks for vector embeddings.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    text_chunks = text_splitter.split_documents(extracted_docs)
    return text_chunks

def download_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Initialize HuggingFace embedding model for vector generation.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings
