import os
import fitz  # PyMuPDF

def extract_bangla_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_text = page.get_text()
            # Filter Bangla Unicode range: U+0980–U+09FF
            bangla_text = ''.join([char for char in page_text if '\u0980' <= char <= '\u09FF' or char.isspace()])
            text += bangla_text + "\n"
    return text

def extract_from_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            bangla_text = extract_bangla_text_from_pdf(pdf_path)
            output_file = os.path.splitext(filename)[0] + '_bangla.txt'
            with open(os.path.join(folder_path, output_file), 'w', encoding='utf-8') as f:
                f.write(bangla_text)

if __name__ == "__main__":
    extract_from_folder('files')