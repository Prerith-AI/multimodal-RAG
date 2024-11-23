# Multimodal RAG Assistant
An advanced Multimodal Retrieval-Augmented Generation (RAG) system that integrates intelligent document processing, semantic search, and powerful language models for answering questions with contextual accuracy. This application supports multiple document formats and employs OCR to extract text from images, making it versatile for real-world use cases.

## Features
Multimodal Document Processing:

Supports PDF, text, Markdown, CSV, Excel, and image-based documents (e.g., PNG, JPG, JPEG).
Uses Tesseract OCR for extracting text from images.
Intelligent Text Chunking:

Dynamically splits text into coherent chunks using RecursiveCharacterTextSplitter for semantic continuity.
Vector Embedding and Search:

Utilizes FAISS for creating vector indexes.
Efficient semantic search for retrieving relevant document chunks.
Language Model Integration:

Supports Claude for text processing and GPT-4o Mini for multimodal tasks.
Employs a robust RAG chain with prompt-based contextual Q&A.
Interactive Web Interface:

Built with Streamlit for an intuitive user experience.
Allows document upload, question input, and real-time AI-driven responses.

## Setup and Installation
### Requirements
Python 3.10 or later
Tesseract OCR installed
Supported libraries: numpy, pandas, langchain, sentence-transformers, faiss, streamlit, dotenv, and opencv-python.
### Installation Steps
Clone the repository:

bash
Copy code
git clone https://github.com/your-repo/multimodal-rag.git
cd multimodal-rag
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Install Tesseract OCR:

Linux:
bash
Copy code
sudo apt update
sudo apt install tesseract-ocr
Windows: Download and install Tesseract OCR from here.
Mac:
bash
Copy code
brew install tesseract
Set up .env file:

Create a .env file in the root directory and add your API keys:
env
Copy code
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key
Run the application:

bash
Copy code
streamlit run MultiModal_RAG.py

## Usage Instructions
Document Upload:

Use the sidebar to upload one or more documents in supported formats (PDF, text, CSV, etc.).
Ask a Question:

Enter a question in the main interface. The system retrieves relevant information from the documents and generates a precise answer.
Results:

View the AI assistant's response and document snippets retrieved for context.

## Architecture
MultimodalDocumentLoader:

Processes various document types and extracts content using appropriate loaders or OCR.
RAGEmbedder:

Converts documents into vector embeddings using a pre-trained SentenceTransformer model.
Enables fast similarity search with FAISS.
RAGLanguageModel:

Configures text and image models for generating context-aware responses.
Creates a RAG pipeline combining context retrieval, prompt formatting, and language model inference.
Streamlit UI:

Offers an intuitive interface for file upload, question input, and result visualization.

## Key Libraries and Frameworks
LangChain: Document processing, embeddings, and AI pipelines.
FAISS: Vector-based semantic search.
Streamlit: Interactive web interface.
Tesseract OCR: Text extraction from image-based documents.

## Future Enhancements
Add support for audio and video file processing.
Implement real-time updates for large document processing.
Enable API endpoint for programmatic access.

Feel free to contribute to this project and share your feedback!




















