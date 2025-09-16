# app.py
import streamlit as st
import os

# Import all custom functions
from src.model.llm_handler import get_rag_chain
from src.ingestion.document_parser import parse_pdf_document
from src.processing.text_processor import process_and_chunk_text
from src.model.embeddings import get_embeddings_and_vector_store, load_vector_store, add_documents_to_vector_store

# Define constants
UPLOAD_DIR = "data/raw"
VECTOR_DB_DIR = "data/vector_db"

def main():
    st.set_page_config(layout="wide")
    st.title("KMRL Document Assistant")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vector_store_exists" not in st.session_state:
        st.session_state["vector_store_exists"] = os.path.exists(os.path.join(VECTOR_DB_DIR, "index.faiss"))

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Document Processing Section ---
    st.sidebar.header("Upload and Process Documents")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a PDF document...",
        type=["pdf"],
        help="Please upload a document to begin processing.",
    )

    if uploaded_file is not None:
        try:
            temp_file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            os.makedirs(VECTOR_DB_DIR, exist_ok=True)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success("File uploaded successfully! Processing...")
            
            # This line will now extract tables and text
            extracted_data = parse_pdf_document(temp_file_path)

            if "Error" in extracted_data:
                st.sidebar.error(f"Failed to process document: {extracted_data}")
            else:
                st.sidebar.success("Document extraction complete! Text and tables processed.")
                
                # Display the extracted data (text and tables)
                with st.expander("View Extracted Data"):
                    st.markdown(extracted_data)
                
                # The rest of the pipeline remains the same
                chunks = process_and_chunk_text(extracted_data)
                
                if chunks:
                    db_exists = os.path.exists(VECTOR_DB_DIR) and os.path.exists(os.path.join(VECTOR_DB_DIR, "index.faiss"))
                    
                    st.sidebar.info(f"Document split into {len(chunks)} chunks. Creating embeddings...")

                    if db_exists:
                        # Add to the existing database
                        vector_store = add_documents_to_vector_store(chunks, db_path=VECTOR_DB_DIR)
                        st.sidebar.success("New documents added to the existing vector database!")
                    else:
                        # Create a brand new database
                        vector_store = get_embeddings_and_vector_store(chunks, db_path=VECTOR_DB_DIR)
                        st.sidebar.success("Embeddings created and stored in a new vector database!")
                        st.session_state["vector_store_exists"] = True

                else:
                    st.sidebar.warning("No text chunks were generated.")
        finally:
            if "temp_file_path" in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    # --- Query and Answer Section ---
    if "vector_store_exists" in st.session_state and st.session_state["vector_store_exists"]:
        if "GROQ_API_KEY" not in st.secrets:
            st.error("Groq API key is not set. Please add it to your Streamlit secrets.")
        else:
            api_token = st.secrets["GROQ_API_KEY"]
            user_query = st.chat_input("Ask a question about the documents:")
            
            if user_query:
                # Add user message to chat history and display it
                st.session_state.messages.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)

                with st.spinner("Generating summary..."):
                    vector_store = load_vector_store(db_path=VECTOR_DB_DIR)
                    qa_chain = get_rag_chain(vector_store, api_token)
                    
                    # Create the chat history format needed by the chain
                    chat_history = []
                    for msg in st.session_state.messages:
                        if msg["role"] == "user":
                            chat_history.append(("user", msg["content"]))
                        else:
                            chat_history.append(("assistant", msg["content"]))

                    # Run the query
                    response = qa_chain({"question": user_query, "chat_history": chat_history})
                    
                    # Display assistant response and add to chat history
                    with st.chat_message("assistant"):
                        st.markdown(response["answer"])
                    
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
    else:
        st.warning("Please upload and process documents first to enable the query functionality.")

if __name__ == "__main__":
    main()