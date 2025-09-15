# src/ingestion/document_parser.py
from pdfminer.high_level import extract_text
import os
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

# Add this line to set the Tesseract command path
# CHANGE THIS PATH IF YOUR TESSERACT INSTALLATION IS DIFFERENT
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def parse_pdf_document(file_path):
    """
    Parses a document to extract all text content, falling back to OCR if needed.

    Args:
        file_path (str): The path to the PDF or image file.

    Returns:
        str: The extracted text from the document.
    """
    if not os.path.exists(file_path):
        return "Error: File not found."

    # Try extracting text from the PDF using pdfminer.six (for text-based PDFs)
    try:
        text = extract_text(file_path)
        if text.strip():  # Check if any text was extracted
            return text
    except Exception as e:
        # Fallback to OCR if the initial text extraction fails
        print(f"Text extraction failed, falling back to OCR: {e}")

    # OCR process for scanned PDFs or images
    try:
        # Convert PDF pages to a list of images
        images = convert_from_path(file_path)
        
        extracted_text = ""
        # Process each page (image) with OCR
        for i, image in enumerate(images):
            # Use pytesseract to extract text from the image
            page_text = pytesseract.image_to_string(image)
            extracted_text += page_text + "\n"
        
        if extracted_text.strip():
            return extracted_text
        else:
            return "Error: OCR did not find any text."
    except Exception as e:
        return f"Error during OCR process: {e}"