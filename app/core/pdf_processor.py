import fitz
import os
from typing import List, Optional
from datetime import datetime

class PDFProcessor:
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)

    async def save_pdf(self, file_content: bytes, filename: str) -> str:
        """Save PDF file to disk and return the file path"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(self.upload_dir, safe_filename)
        
        with open(file_path, "wb") as pdf_file:
            pdf_file.write(file_content)
        
        return file_path

    def extract_text(self, file_path: str) -> List[str]:
        """Extract text from PDF and return it as chunks"""
        try:
            doc = fitz.open(file_path)
            chunks = []
            
            for page in doc:
                text = page.get_text()
                # Simple chunking by page, you might want to implement more sophisticated chunking
                if text.strip():
                    chunks.append(text)
            
            doc.close()
            return chunks
        
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            raise

    async def process_pdf(self, file_content: bytes, filename: str) -> tuple[str, List[str]]:
        """Process PDF file and return file path and extracted text chunks"""
        file_path = await self.save_pdf(file_content, filename)
        text_chunks = self.extract_text(file_path)
        return file_path, text_chunks