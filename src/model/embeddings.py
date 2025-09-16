# src/model/embeddings.py
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os

def get_embeddings_and_vector_store(text_chunks, db_path="data/vector_db"):
    """
    Creates a brand-new vector store from a list of text chunks.
    This is used for the very first document upload.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local(db_path)
    return vector_store

def load_vector_store(db_path="data/vector_db"):
    """
    Loads an existing FAISS vector store from a local path.
    The 'allow_dangerous_deserialization=True' flag is set to enable loading
    the vector store from a local file, which we trust in this case.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

def add_documents_to_vector_store(text_chunks, db_path="data/vector_db"):
    """
    Adds new document embeddings to an existing vector store.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load the existing vector store
    existing_vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    
    # Add the new documents to the existing store
    existing_vector_store.add_texts(text_chunks)
    
    # Save the updated vector store
    existing_vector_store.save_local(db_path)

    return existing_vector_store