FROM python:3.10-slim

# Install Tesseract with Bengali language + other system packages
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-ben \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and files
COPY app/ ./app
COPY files/ ./files

#Expose the port for Streamlit
EXPOSE 8000

# Create a healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8000/_stcore/health

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8000
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run the Streamlit chatbot
CMD ["streamlit", "run", "app/brta_chatbot.py", "--server.port=8000", "--server.address=0.0.0.0"]

#File Processor
# CMD ["python", "app/process_files.py"]