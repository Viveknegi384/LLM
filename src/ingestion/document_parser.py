# src/ingestion/document_parser.py
from pdfminer.high_level import extract_text
import os

def parse_pdf_document(file_path):
    """
    Parses a PDF document to extract all text content.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF.
    """
    if not os.path.exists(file_path):
        return "Error: File not found."

    try:
        # Use pdfminer.six's high-level API to extract all text
        text = extract_text(file_path)
        return text
    except Exception as e:
        return f"Error parsing PDF: {e}"