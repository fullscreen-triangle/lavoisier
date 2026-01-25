#!/usr/bin/env python3
"""Extract text from PDF file."""

import sys
import os

def extract_pdf_text(pdf_path):
    """Extract text from PDF using available library."""
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ''.join([page.extract_text() for page in reader.pages])
            return text
    except ImportError:
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            text = ''.join([page.get_text() for page in doc])
            doc.close()
            return text
        except ImportError:
            try:
                from pdfminer.high_level import extract_text
                return extract_text(pdf_path)
            except ImportError:
                print("ERROR: No PDF extraction library available (PyPDF2, PyMuPDF, or pdfminer)")
                return None

if __name__ == '__main__':
    pdf_path = os.path.join('sources', 'hardware-based-temporal-measurements.pdf')
    if not os.path.exists(pdf_path):
        pdf_path = 'hardware-based-temporal-measurements.pdf'
    
    text = extract_pdf_text(pdf_path)
    if text:
        print(text)
    else:
        sys.exit(1)
