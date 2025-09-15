# src/processing/text_processor.py
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_and_chunk_text(text):
    """
    Splits a long text document into smaller, overlapping chunks.

    Args:
        text (str): The full text content of a document.

    Returns:
        list: A list of text chunks (strings).
    """
    # Initialize the RecursiveCharacterTextSplitter
    # You can adjust the chunk_size and chunk_overlap as needed
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Each chunk will have up to 1000 characters
        chunk_overlap=200,  # Chunks will overlap by 200 characters to maintain context
        length_function=len,
        add_start_index=True,
    )

    # Split the text into chunks
    chunks = text_splitter.create_documents([text])
    
    # Return the chunks as a list of strings
    return [chunk.page_content for chunk in chunks]