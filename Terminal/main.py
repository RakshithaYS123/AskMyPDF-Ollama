import os
import sys
import time
import logging
from pathlib import Path

# Set Ollama API host to match the running server
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11500"

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def main():
    # Check for correct usage
    if len(sys.argv) < 2:
        logging.error("Usage: python main.py <PDF_FILE_PATH>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        logging.error(f"File not found: {pdf_path}")
        sys.exit(1)

    # 1. Initialize the model and get working embeddings
    logging.info("Loading models...")
    
    # Retry LLM initialization with backoff
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Use explicit timeout and request parameters
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
                sys.exit(1)
    
    # Get working embeddings model
    try:
        embeddings = get_embeddings_model()
    except Exception as e:
        logging.error(f"Could not initialize any embedding model: {e}")
        sys.exit(1)

    # 2. Load the PDF file
    logging.info(f"Loading PDF: {pdf_path}")
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        if not pages:
            logging.error("No content extracted from PDF")
            sys.exit(1)
        logging.info(f"Loaded {len(pages)} pages from PDF")
    except Exception as e:
        logging.error(f"Error loading PDF: {e}")
        sys.exit(1)

    # 3. Set up or load FAISS index
    pdf_name = Path(pdf_path).stem
    index_path = f"faiss_index_{pdf_name}"
    
    try:
        if os.path.exists(index_path):
            logging.info("Loading existing FAISS index...")
            try:
                store = FAISS.load_local(index_path, embeddings)
                logging.info("Successfully loaded existing index")
            except Exception as load_error:
                logging.warning(f"Could not load existing index: {load_error}. Creating new index...")
                store = FAISS.from_documents(pages, embeddings)
                store.save_local(index_path)
        else:
            logging.info("Creating new FAISS index...")
            store = FAISS.from_documents(pages, embeddings)
            store.save_local(index_path)
            logging.info(f"Saved index to {index_path}")
    except Exception as e:
        logging.error(f"Error with FAISS index: {e}")
        sys.exit(1)

    # Create retriever with specific parameters for better results
    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3, "fetch_k": 5}  # Fetch more documents but return top 3
    )

    # 4. Define the prompt template with clear instructions
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

    def format_docs(docs):
        formatted = []
        for i, doc in enumerate(docs):
            formatted.append(f"Excerpt {i+1}:\n{doc.page_content}")
        return "\n\n".join(formatted)

    # 5. Build the chain with direct error handling
    def process_query(question):
        try:
            # Get relevant documents
            docs = retriever.invoke(question)
            if not docs:
                return "No relevant information found in the document."
            
            # Format context
            context = format_docs(docs)
            logging.info(f"Retrieved {len(docs)} relevant document sections")
            
            # Prepare prompt input
            prompt_input = {
                "context": context,
                "question": question
            }
            
            # Get prompt text
            prompt_text = prompt.format(**prompt_input)
            logging.info(f"Sending prompt with {len(prompt_text)} characters to LLM")
            
            # Get LLM response
            response = llm.invoke(prompt_text)
            logging.info(f"Received response with {len(response)} characters")
            
            return response
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return f"Error processing your question: {str(e)}"

    # 6. Start the interactive loop
    logging.info(f"Ready! Ask questions about {pdf_path}")
    print("\n" + "="*50)
    print(f"PDF QA System for: {pdf_path}")
    print("="*50 + "\n")
    
    while True:
        try:
            question = input("\nQuestion (type 'exit' to quit): ")
            if question.lower() in ("exit", "quit", "q"):
                logging.info("Exiting program.")
                break
            
            print("\nSearching document...\n")
            
            # Use direct function instead of chain to better debug
            answer = process_query(question)
            
            print(f"Answer: {answer}\n")
        
        except KeyboardInterrupt:
            logging.info("Exiting due to keyboard interrupt...")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()