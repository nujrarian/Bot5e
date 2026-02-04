import fitz  # PyMuPDF
import re
from logger import setup_logger

logger = setup_logger(__name__)

def clean_text(text):
    """Clean extracted PDF text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove page headers/footers
    text = re.sub(r'Not\s+for\s+resale.*?System\s+Reference\s+Document\s+5\.1\s+\d+', '', text)
    # Fix common OCR issues
    text = text.replace('–', '-')
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    return text.strip()

def pdf_read_split(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    Load a PDF using PyMuPDF and split it into clean chunks.

    Args:
        pdf_path: Path to the PDF file
        chunk_size: Maximum size of each chunk (in characters)
        chunk_overlap: Overlap between chunks

    Returns:
        List of text chunks

    Raises:
        FileNotFoundError: If PDF file doesn't exist
        ValueError: If PDF is empty or invalid
    """
    logger.info(f"Extracting text from {pdf_path}...")

    try:
        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        full_text = ""

        # Extract text from each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            full_text += page_text + "\n\n"

        doc.close()

        logger.info(f"Extracted {len(full_text)} characters")

        if not full_text.strip():
            raise ValueError("PDF appears to be empty or contains no extractable text")

        # Clean the text
        full_text = clean_text(full_text)
        logger.info(f"After cleaning: {len(full_text)} characters")

        # Split into chunks by sentences to avoid breaking mid-sentence
        sentences = re.split(r'(?<=[.!?])\s+', full_text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If adding this sentence exceeds chunk_size, save current chunk
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap (use last N characters for overlap)
                if len(current_chunk) > chunk_overlap:
                    overlap_text = current_chunk[-chunk_overlap:]
                else:
                    overlap_text = current_chunk
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk += " " + sentence

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        if not chunks:
            raise ValueError("Failed to create any text chunks from PDF")

        logger.info(f"Created {len(chunks)} chunks")
        logger.info(f"Average chunk size: {sum(len(c) for c in chunks) / len(chunks):.0f} characters")

        return chunks

    except FileNotFoundError:
        logger.error(f"PDF file not found: {pdf_path}")
        raise
    except fitz.fitz.FileDataError as e:
        logger.error(f"Invalid or corrupted PDF file: {e}")
        raise ValueError(f"Invalid PDF file: {pdf_path}")
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise