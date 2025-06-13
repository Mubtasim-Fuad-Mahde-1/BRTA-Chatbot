# BRTA-Chatbot

This is a new BRTA chatbot mvp system. Later this will be integrated for any government website

.env

PINECONE_API_KEY=pcsk_4gAiEB_QfGhTJGc7LMhu6mYaUeRf2bYrvN2e8BMc8yeXV6NEZ44njrUPsCxt78nxpeoyfw

INDEX_NAME=brta-index

GOOGLE_API_KEY=AIzaSyDVw5eA-bGO2F_drHmZZb-yeMEYcySm5bs

CHAT_MODEL=gemini/gemini-2.5-flash-preview-05-20

EMBED_MODEL=models/text-embedding-004

VECTOR_DIM=768

# How to run file processor?

1. Install docker
2. In terminal run this command - docker build -t bangla-extractor .
3. Then run this command - docker run --rm -it --env-file .env -p 8000:8000 bangla-extractor

# How to run chatbot?

1. Install docker desktop
2. in terminal run this command - docker build -t brta-chatbot .
3. Then run this command - docker run --rm -it --env-file .env -p 8000:8000 brta-chatbot

# How much progress made-

* processing pdf and text file complete
* uploading processed pdf and text file to vector db complete
* Chatbot integration with tool and streamlit ui complete

# To do:

1. collect more files and data from BRTA website
