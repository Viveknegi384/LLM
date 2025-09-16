# src/ingestion/document_parser.py
import pdfplumber
import os
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

# Add this line to set the Tesseract command path
# CHANGE THIS PATH IF YOUR TESSERACT INSTALLATION IS DIFFERENT
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def parse_pdf_document(file_path):
    """
    Parses a document to extract all text and tables, using OCR as a fallback.

    Args:
        file_path (str): The path to the PDF or image file.

    Returns:
        str: The extracted text from the document.
    """
    if not os.path.exists(file_path):
        return "Error: File not found."

    combined_content = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            # We'll use a loop to process page by page
            for i, page in enumerate(pdf.pages):
                page_content = ""
                
                # --- Attempt 1: Extract text and tables using pdfplumber ---
                page_text = page.extract_text()
                if page_text:
                    page_content += page_text + "\n\n"

                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        markdown_table = "Table on Page " + str(i + 1) + ":\n"
                        header = table[0]
                        rows = table[1:]
                        markdown_table += "| " + " | ".join(map(str, header)) + " |\n"
                        markdown_table += "|---" * len(header) + "|\n"
                        for row in rows:
                            markdown_table += "| " + " | ".join(map(str, row)) + " |\n"
                        page_content += markdown_table + "\n\n"

                # --- Attempt 2: Fallback to OCR if nothing was found ---
                if not page_content.strip():
                    try:
                        # Convert the current page to an image
                        image = convert_from_path(file_path, first_page=i+1, last_page=i+1)[0]
                        # Use pytesseract to extract text from the image
                        page_text_ocr = pytesseract.image_to_string(image)
                        if page_text_ocr.strip():
                            page_content = page_text_ocr + "\n\n"
                        else:
                            page_content = "OCR found no text on this page.\n\n"
                    except Exception as ocr_e:
                        page_content = f"Error during OCR on page {i+1}: {ocr_e}\n\n"

                combined_content += page_content
    except Exception as e:
        return f"Error processing document: {e}"

    if not combined_content.strip():
        return "Error: Document appears to be empty or unreadable."

    return combined_content