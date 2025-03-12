# AskMyPDF-Ollama
# PDF Question Answering System

A Python-based application for asking questions about PDF documents using AI. This repository contains two implementations:

1. **Flask Web Application** - A web interface for uploading PDFs and interacting with them through a chat-like interface
2. **Terminal Application** - A command-line interface for the same functionality

## Overview

This system uses the Ollama API to power a question-answering service that allows users to upload PDF documents and ask natural language questions about their content. The application:

1. Extracts text from PDF documents
2. Creates embeddings and vector stores for efficient retrieval
3. Uses language models to answer questions based on document content

## Requirements

- Python 3.8+
- Ollama server running locally (default: http://127.0.0.1:11502)
- Required Python packages (see Installation)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/pdf-qa-system.git
   cd pdf-qa-system
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure Ollama server is running with the following models:
   - LLM: `llama3`
   - Embeddings: `nomic-embed-text` (falls back to `all-minilm` or `znbang/bge:small-en-v1.5-f32`)

## Flask Web Application

### Directory Structure
```
Flask_Application/
├── app.py            # Main Flask application
├── templates/
│   ├── index.html    # Home page for PDF uploads
│   └── pdf.html      # QA interface for specific PDFs
└── uploads/          # Storage for uploaded PDFs (created automatically)
```

### Features
- Clean, modern UI with Bootstrap
- Drag-and-drop PDF uploads
- Responsive chat interface for questions and answers
- Real-time feedback during processing

### Running the Web App
```bash
cd Flask_Application
python app.py
```

Then open your browser and navigate to `http://localhost:5000`

## Terminal Application

The terminal application provides similar functionality to the web app but through a command-line interface.

### Running the Terminal App
```bash
cd terminal
python main.py path/to/your/document.pdf
```

Follow the prompts to ask questions about the loaded document.

## How It Works

1. **PDF Processing**:
   - PDFs are loaded and split into chunks
   - Each chunk is converted to vector embeddings
   - Vectors are stored in a FAISS index for efficient retrieval

2. **Question Answering**:
   - User submits a question
   - System retrieves the most relevant document chunks
   - The LLM generates an answer based on retrieved chunks
   - Answer is presented to the user

## Troubleshooting

- **Model Initialization Errors**: Ensure Ollama server is running and the required models are available
- **PDF Processing Issues**: Make sure uploaded PDFs are not corrupted and contain extractable text
- **Performance Issues**: Large PDFs may take longer to process; consider optimizing chunk size

## License

[MIT License](LICENSE)

## Acknowledgments

- This project uses LangChain for the RAG (Retrieval Augmented Generation) pipeline
- UI components powered by Bootstrap and Font Awesome
