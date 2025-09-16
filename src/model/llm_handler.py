# src/model/llm_handler.py
from langchain.chains import ConversationalRetrievalChain
from groq import Groq
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import streamlit as st
import os

def get_rag_chain(vector_store, api_token):
    # This is an older, less reliable model.
    # It is recommended to use the new one below
    # llm = ChatGroq(api_key=api_token, model_name="llama3-8b-8192")
    
    # Use the ConversationalRetrievalChain, which handles chat history
    
    # --- The updated prompt template with a more forceful instruction ---
    
    prompt_template = """
    You are a helpful and expert assistant for the Kochi Metro Rail Limited (KMRL).
    Your task is to answer user questions using ONLY the provided context.
    
    If the answer is NOT found in the context, your response MUST be ONLY: "I do not have the information in my knowledge base."
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    
    llm = ChatGroq(api_key=api_token, model_name="llama-3.1-8b-instant")

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        combine_docs_chain_kwargs={
            "prompt": PromptTemplate.from_template(prompt_template)
        }
    )
    
    return qa_chain