import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import uuid
import hashlib
# Load environment variables
load_dotenv()

import google.generativeai as genai


# Set Tesseract language to Bengali (ben)
TESSERACT_LANG = 'ben+eng'  # Use 'ben' for Bengali and 'eng' for English
FILES_DIR = 'files'


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
        
    def upsert(self, data):
        """Upsert data (dictionary) into the Pinecone index."""
        vectors = []
        for file, text in data.items():
            if not text.strip():
                print(f"Skipping empty file: {file}")
                continue
            embedding = self.embed_text(text)
            metadata = {'text': text}
            text_id = hashlib.md5(text[:100].encode('utf-8')).hexdigest()
            vectors.append((
                text_id,  # Generate a unique ID for each document
                embedding,
                metadata
            ))
        self.index.upsert(vectors=vectors) # Uncomment this line to actually upsert
        print(f'Upserted {len(vectors)} vectors into Pinecone index: {self.index_name}')
        
    def embed_text(self, text):
        """Embed text using the configured model."""
        response = genai.embed_content(
            model=self.embed_model,
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return response["embedding"]


#============================================
# File processing class for extracting text from PDF and TXT files
#============================================
class FileProcessing:
    def __init__(self):
        self.folder_path = FILES_DIR
        self.TESSERACT_LANG = TESSERACT_LANG
        
    def extract_text_from_pdf(self, pdf_path):
        images = convert_from_path(pdf_path, dpi=300)
        text = ''
        for img in images:
            text += pytesseract.image_to_string(img, lang=TESSERACT_LANG, config='--psm 6')
        return text
    
    def extract_text_from_txt(self, text_path):
        with open(text_path, 'r', encoding='utf-8') as file:
            return file.read()

    def process_folder(self):
        results = {}
        for filename in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, filename)
            if filename.endswith('.pdf'):
                text = self.extract_text_from_pdf(file_path)
                results[filename] = text
            elif filename.endswith('.txt'):
                text = self.extract_text_from_txt(file_path)
                results[filename] = text
            else:
                print(f"Unsupported file type: {filename}")
        if not results:
            print("No files processed. Please check the folder path and file types.")
        else:
            print(f"Processed {len(results)} files.")
        return results

if __name__ == '__main__':
    file_processor = FileProcessing()
    pinecone_client = PineconeClient()
    
    # Process files in the specified folder
    processed_data = file_processor.process_folder()
    
    # Upsert the processed data into Pinecone
    pinecone_client.upsert(processed_data)
    
    print("File processing and upsert completed successfully.")