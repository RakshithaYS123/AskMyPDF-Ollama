import os
import sys
import time
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename

# Set Ollama API host to match the running server
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11502"

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Flask app configuration
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store models and data
llm = None
embeddings = None
pdf_data = {}  # Dictionary to store PDF data: {pdf_id: {'path': path, 'store': FAISS store}}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_embeddings_model():
    """Try different embedding models in order of preference"""
    models_to_try = [
        "nomic-embed-text",  # Try this first (more reliable)
        "all-minilm",        # Another alternative
        "znbang/bge:small-en-v1.5-f32"  # Try this as fallback
    ]
    
    for model_name in models_to_try:
        try:
            logging.info(f"Trying embedding model: {model_name}")
            embeddings = OllamaEmbeddings(model=model_name)
            # Test with a simple embedding
            test_embedding = embeddings.embed_query("test")
            if test_embedding and len(test_embedding) > 0:
                logging.info(f"Successfully using embedding model: {model_name}")
                return embeddings
        except Exception as e:
            logging.warning(f"Failed to use embedding model {model_name}: {e}")
    
    raise RuntimeError("All embedding models failed. Please check Ollama server status.")

def initialize_models():
    """Initialize the LLM and embedding models"""
    global llm, embeddings
    
    # Initialize LLM with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            llm = OllamaLLM(
                model="llama3",
                temperature=0.1,
                request_timeout=60.0,
                stop=["Human:", "Assistant:"],
                num_predict=2048,
            )
            # Test the LLM with a simple query
            test_response = llm.invoke("Hello, are you working?")
            logging.info(f"LLM test response received: {len(test_response)} characters")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logging.warning(f"LLM initialization failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logging.error(f"Failed to initialize LLM after {max_retries} attempts")
                return False
    
    # Initialize embeddings model
    try:
        embeddings = get_embeddings_model()
        return True
    except Exception as e:
        logging.error(f"Could not initialize any embedding model: {e}")
        return False

def load_pdf(pdf_path):
    """Load a PDF and create/load FAISS index"""
    pdf_name = Path(pdf_path).stem
    pdf_id = secure_filename(pdf_name)
    index_path = f"faiss_index_{pdf_id}"
    
    try:
        # Load the PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        if not pages:
            logging.error("No content extracted from PDF")
            return None
        
        logging.info(f"Loaded {len(pages)} pages from PDF")
        
        # Set up or load FAISS index
        if os.path.exists(index_path):
            try:
                store = FAISS.load_local(index_path, embeddings)
                logging.info("Successfully loaded existing index")
            except Exception as load_error:
                logging.warning(f"Could not load existing index: {load_error}. Creating new index...")
                store = FAISS.from_documents(pages, embeddings)
                store.save_local(index_path)
        else:
            store = FAISS.from_documents(pages, embeddings)
            store.save_local(index_path)
        
        # Save in our document store
        pdf_data[pdf_id] = {
            'path': pdf_path,
            'store': store,
            'name': pdf_name
        }
        
        return pdf_id
        
    except Exception as e:
        logging.error(f"Error processing PDF: {e}")
        return None

def process_query(pdf_id, question):
    """Process a query against a specific PDF"""
    if pdf_id not in pdf_data:
        return "PDF not found or not loaded correctly."
    
    try:
        # Get the store
        store = pdf_data[pdf_id]['store']
        
        # Create retriever
        retriever = store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3, "fetch_k": 5}  # Fetch more documents but return top 3
        )
        
        # Define the prompt template
        template = """
        You are a helpful assistant that answers questions based on the provided document excerpts.
        
        Document excerpts:
        {context}
        
        Question: {question}
        
        Important: Answer the question based ONLY on the information in the document excerpts. 
        If the answer is not in the excerpts, say "I don't see information about that in the document."
        Keep your answer concise and to the point.
        
        Answer:
        """
        
        prompt = PromptTemplate.from_template(template)
        
        # Format documents function
        def format_docs(docs):
            formatted = []
            for i, doc in enumerate(docs):
                formatted.append(f"Excerpt {i+1}:\n{doc.page_content}")
            return "\n\n".join(formatted)
        
        # Get relevant documents
        docs = retriever.invoke(question)
        if not docs:
            return "No relevant information found in the document."
        
        # Format context
        context = format_docs(docs)
        
        # Prepare prompt input
        prompt_input = {
            "context": context,
            "question": question
        }
        
        # Get prompt text
        prompt_text = prompt.format(**prompt_input)
        
        # Get LLM response
        response = llm.invoke(prompt_text)
        
        return response
    
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return f"Error processing your question: {str(e)}"

# Initialize models before the first request
@app.route('/init')
def init_models():
    """Initialize models and return status"""
    global llm, embeddings
    if llm is None or embeddings is None:
        if initialize_models():
            return jsonify({'status': 'success', 'message': 'Models initialized successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to initialize models'})
    return jsonify({'status': 'success', 'message': 'Models already initialized'})

# Routes
@app.route('/')
def index():
    """Home page route"""
    # Get list of loaded PDFs
    loaded_pdfs = [{'id': pdf_id, 'name': data['name']} for pdf_id, data in pdf_data.items()]
    return render_template('index.html', pdfs=loaded_pdfs)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle PDF file upload"""
    # Ensure models are initialized
    global llm, embeddings
    if llm is None or embeddings is None:
        if not initialize_models():
            return jsonify({'status': 'error', 'message': 'Failed to initialize models'})
    
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Load PDF and create index
        pdf_id = load_pdf(file_path)
        
        if pdf_id:
            return jsonify({
                'status': 'success', 
                'message': 'File uploaded and processed successfully',
                'pdf_id': pdf_id,
                'pdf_name': Path(file_path).stem
            })
        else:
            return jsonify({'status': 'error', 'message': 'Failed to process PDF'})
    
    return jsonify({'status': 'error', 'message': 'Invalid file type'})

@app.route('/ask', methods=['POST'])
def ask_question():
    """Process a question about a PDF"""
    # Ensure models are initialized
    global llm, embeddings
    if llm is None or embeddings is None:
        if not initialize_models():
            return jsonify({'status': 'error', 'message': 'Failed to initialize models'})
    
    data = request.json
    pdf_id = data.get('pdf_id')
    question = data.get('question')
    
    if not pdf_id or not question:
        return jsonify({'status': 'error', 'message': 'Missing PDF ID or question'})
    
    if pdf_id not in pdf_data:
        return jsonify({'status': 'error', 'message': 'PDF not found'})
    
    # Process the query
    answer = process_query(pdf_id, question)
    
    return jsonify({
        'status': 'success',
        'answer': answer,
        'pdf_name': pdf_data[pdf_id]['name']
    })

@app.route('/pdf/<pdf_id>')
def pdf_page(pdf_id):
    """Page for asking questions about a specific PDF"""
    if pdf_id not in pdf_data:
        return redirect(url_for('index'))
    
    return render_template('pdf.html', 
                          pdf_id=pdf_id, 
                          pdf_name=pdf_data[pdf_id]['name'])

@app.route('/status')
def system_status():
    """Check the status of the LLM and embeddings models"""
    return jsonify({
        'llm_loaded': llm is not None,
        'embeddings_loaded': embeddings is not None,
        'pdfs_loaded': len(pdf_data)
    })

if __name__ == '__main__':
    # Initialize models at startup
    if initialize_models():
        logging.info("Models initialized successfully")
        app.run(debug=True, port=5000)
    else:
        logging.error("Failed to initialize models. Exiting.")
        sys.exit(1)