import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import uuid
import hashlib
import logging

# Load environment variables
load_dotenv()

import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Tesseract language to Bengali (ben)
TESSERACT_LANG = 'ben+eng'  # Use 'ben' for Bengali and 'eng' for English
FILES_DIR = 'files'

# Increase PIL's image size limit or remove it entirely
Image.MAX_IMAGE_PIXELS = None  # Remove limit entirely
# Alternative: Image.MAX_IMAGE_PIXELS = 300000000  # Set higher limit

#============================================
# Pinecone client for managing vector database operations
#============================================
class PineconeClient:
    def __init__(self):
        self.api_key = os.getenv('PINECONE_API_KEY')
        self.index_name = os.getenv('INDEX_NAME')
        self.vector_dim = int(os.getenv('VECTOR_DIM'))
        self.pc = Pinecone(api_key=self.api_key)
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        if self.index_name not in self.pc.list_indexes():
            try:
                self.pc.create_index(
                    name=self.index_name, 
                    dimension=self.vector_dim,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
            except Exception as e:
                if "ALREADY_EXISTS" not in str(e):
                    raise
                else:
                    print(f"Index '{self.index_name}' already exists, continuing.")
        self.index = self.pc.Index(self.index_name)
        self.embed_model = os.getenv('EMBED_MODEL')
    
    def chunk_text(self, text, chunk_size=2000, overlap=300):
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at word boundary
            if end < len(text) and not text[end].isspace():
                last_space = chunk.rfind(' ')
                if last_space > start + chunk_size // 2:  # Only if we find a space reasonably close
                    chunk = text[start:start + last_space]
                    end = start + last_space
            
            chunks.append(chunk.strip())
            start = end - overlap
            
        return [chunk for chunk in chunks if chunk.strip()]
        
    def upsert(self, data):
        """Upsert data (dictionary) into the Pinecone index with chunking."""
        vectors = []
        for file, text in data.items():
            if not text.strip():
                logger.warning(f"Skipping empty file: {file}")
                continue
                
            # Chunk the text
            chunks = self.chunk_text(text)
            logger.info(f"Processing {file}: {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks):
                try:
                    embedding = self.embed_text(chunk)
                    metadata = {
                        'text': chunk,
                        'source_file': file,
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    }
                    # Create unique ID for each chunk
                    chunk_id = f"{hashlib.md5(file.encode()).hexdigest()}_{i}"
                    vectors.append((chunk_id, embedding, metadata))
                except Exception as e:
                    logger.error(f"Failed to embed chunk {i} from {file}: {e}")
                    continue
        
        if vectors:
            # Upsert in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                try:
                    self.index.upsert(vectors=batch)
                    logger.info(f"Upserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
                except Exception as e:
                    logger.error(f"Failed to upsert batch: {e}")
            
            print(f'Successfully upserted {len(vectors)} vectors into Pinecone index: {self.index_name}')
        else:
            logger.warning("No vectors to upsert")
        
    def embed_text(self, text):
        """Embed text using the configured model."""
        try:
            response = genai.embed_content(
                model=self.embed_model,
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            return response["embedding"]
        except Exception as e:
            logger.error(f"Embedding failed for text length {len(text)}: {e}")
            raise


#============================================
# File processing class for extracting text from PDF and TXT files
#============================================
class FileProcessing:
    def __init__(self):
        self.folder_path = FILES_DIR
        self.TESSERACT_LANG = TESSERACT_LANG
        
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF with progressive DPI fallback."""
        dpi_options = [200, 150, 100, 75]  # Try different DPI values
        
        for dpi in dpi_options:
            try:
                logger.info(f"Attempting PDF conversion with DPI: {dpi}")
                images = convert_from_path(pdf_path, dpi=dpi)
                text = ''
                
                for page_num, img in enumerate(images):
                    try:
                        page_text = pytesseract.image_to_string(
                            img, 
                            lang=self.TESSERACT_LANG, 
                            config='--psm 6'
                        )
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as e:
                        logger.warning(f"OCR failed for page {page_num + 1}: {e}")
                        continue
                
                logger.info(f"Successfully processed PDF with DPI {dpi}")
                return text
                
            except Exception as e:
                logger.warning(f"Failed with DPI {dpi}: {e}")
                if dpi == dpi_options[-1]:  # Last option
                    logger.error(f"All DPI options failed for {pdf_path}")
                    return f"ERROR: Could not process PDF {pdf_path}: {str(e)}"
                continue
        
        return ""
    
    def extract_text_from_txt(self, text_path):
        """Extract text from TXT file with error handling."""
        try:
            with open(text_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try different encodings
            encodings = ['latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    with open(text_path, 'r', encoding=encoding) as file:
                        logger.info(f"Successfully read {text_path} with {encoding} encoding")
                        return file.read()
                except UnicodeDecodeError:
                    continue
            logger.error(f"Could not decode {text_path} with any encoding")
            return f"ERROR: Could not decode file {text_path}"
        except Exception as e:
            logger.error(f"Error reading {text_path}: {e}")
            return f"ERROR: Could not read file {text_path}: {str(e)}"

    def process_folder(self):
        """Process all files in the folder with comprehensive error handling."""
        results = {}
        
        if not os.path.exists(self.folder_path):
            logger.error(f"Folder {self.folder_path} does not exist")
            return results
            
        files = os.listdir(self.folder_path)
        logger.info(f"Found {len(files)} files to process")
        
        for filename in files:
            file_path = os.path.join(self.folder_path, filename)
            
            # Skip directories
            if os.path.isdir(file_path):
                continue
                
            logger.info(f"Processing: {filename}")
            
            try:
                if filename.lower().endswith('.pdf'):
                    text = self.extract_text_from_pdf(file_path)
                    results[filename] = text
                elif filename.lower().endswith('.txt'):
                    text = self.extract_text_from_txt(file_path)
                    results[filename] = text
                else:
                    logger.warning(f"Unsupported file type: {filename}")
                    continue
                    
                # Log text length for debugging
                logger.info(f"Extracted {len(text)} characters from {filename}")
                
            except Exception as e:
                logger.error(f"Failed to process {filename}: {e}")
                results[filename] = f"ERROR: Failed to process {filename}: {str(e)}"
                continue
        
        if not results:
            logger.warning("No files processed successfully")
        else:
            logger.info(f"Successfully processed {len(results)} files")
            
        return results

if __name__ == '__main__':
    try:
        file_processor = FileProcessing()
        pinecone_client = PineconeClient()
        
        # Process files in the specified folder
        processed_data = file_processor.process_folder()
        
        if processed_data:
            # Upsert the processed data into Pinecone
            pinecone_client.upsert(processed_data)
            print("File processing and upsert completed successfully.")
        else:
            print("No data to process.")
            
    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise