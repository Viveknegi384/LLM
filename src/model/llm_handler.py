# src/model/llm_handler.py
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

def get_rag_chain(vector_store, api_token):
    """
    Creates a Retrieval-Augmented Generation (RAG) chain that is compatible
    with conversational-task-only providers.
    """
    # Create a custom prompt template to add the specific instruction
    # The instruction is to only answer if the information is in the context
    prompt_template = """
    You are a helpful and expert assistant for the Kochi Metro Rail Limited (KMRL). 
    Your task is to answer user questions using only the provided context.
    
    If the answer is not found in the context, clearly state that you do not have the information in your knowledge base.
    
    Context:
    {context}
    
    Chat History:
    {chat_history}
    
    Question:
    {question}
    
    Answer:
    """

    # Initialize the LLM with the HuggingFaceEndpoint
    llm = HuggingFaceEndpoint(
        repo_id="google/gemma-2-9b-it",
        huggingfacehub_api_token=api_token,
        temperature=0.1,
        max_new_tokens=512,
    )

    # Wrap the LLM in a ChatHuggingFace model to ensure compatibility
    chat_model = ChatHuggingFace(llm=llm)

    # Use the ConversationalRetrievalChain, which is designed for this task
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vector_store.as_retriever(),
        combine_docs_chain_kwargs={
            "prompt": PromptTemplate.from_template(prompt_template)
        }
    )
    
    return qa_chain