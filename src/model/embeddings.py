# src/model/embeddings.py
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os

def get_embeddings_and_vector_store(text_chunks, db_path="faiss_index"):
    """
    Creates embeddings from text chunks and stores them in a FAISS vector database.

    Args:
        text_chunks (list): A list of text chunks (strings).
        db_path (str): The path to save the FAISS database.

    Returns:
        FAISS: The initialized FAISS vector store.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local(db_path)
    return vector_store

def load_vector_store(db_path="faiss_index"):
    """
    Loads an existing FAISS vector store from a local path.
    The 'allow_dangerous_deserialization=True' flag is set to enable loading
    the vector store from a local file, which we trust in this case.

    Args:
        db_path (str): The path to the saved FAISS database.

    Returns:
        FAISS: The loaded FAISS vector store.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)