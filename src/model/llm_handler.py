# src/model/llm_handler.py
from langchain.chains import ConversationalRetrievalChain
from groq import Groq
from langchain_groq import ChatGroq  # Correct import for the Runnable Groq model
import streamlit as st
import os

def get_rag_chain(vector_store, api_token):
    """
    Creates a Retrieval-Augmented Generation (RAG) chain that is compatible
    with the Groq LLM.
    """
    # Initialize the Groq LLM as a Runnable
    # Note: Using ChatGroq which is a Runnable and works with new LangChain
    llm = ChatGroq(api_key=api_token, model_name="llama-3.1-8b-instant")
    
    # Use the ConversationalRetrievalChain, which handles chat history
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
    )
    
    return qa_chain