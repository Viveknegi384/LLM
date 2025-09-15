# src/model/llm_handler.py
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import ChatHuggingFace
# Change this import
from langchain_huggingface import HuggingFaceEndpoint
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def get_rag_chain(vector_store, api_token):
    """
    Creates a Retrieval-Augmented Generation (RAG) chain that is compatible
    with conversational-task-only providers.
    """
    # Initialize the LLM with the HuggingFaceEndpoint
    llm = HuggingFaceEndpoint(
        repo_id="google/gemma-2-9b-it",
        huggingfacehub_api_token=api_token,
        temperature=0.1,
        max_new_tokens=512,
    )
    
    # Wrap the LLM in a ChatHuggingFace model to ensure compatibility with conversational tasks.
    chat_model = ChatHuggingFace(llm=llm)

    # Use the ConversationalRetrievalChain, which is designed for this task
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vector_store.as_retriever(),
    )
    
    return qa_chain