# Multimodal RAG Assistant
An advanced Multimodal Retrieval-Augmented Generation (RAG) system that integrates intelligent document processing, semantic search, and powerful language models for answering questions with contextual accuracy. This application supports multiple document formats and employs OCR to extract text from images, making it versatile for real-world use cases.

## Key Features
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

![Screenshot 2024-11-23 171141](https://github.com/user-attachments/assets/a374edcd-0915-4e5c-9d88-4dafb8e099e4)



Install dependencies:


![Screenshot 2024-11-23 171242](https://github.com/user-attachments/assets/82e604b8-96c6-4296-b754-e4046207a6af)


Install Tesseract OCR:

Linux:


![Screenshot 2024-11-23 171338](https://github.com/user-attachments/assets/e4fc8eb7-af42-4514-a265-d904b38a5f25)


Windows: Download and install Tesseract OCR from https://github.com/tesseract-ocr/tesseract.git

Mac:


![Screenshot 2024-11-23 171512](https://github.com/user-attachments/assets/fba0f309-b1a4-40c8-99ca-21dbcfca0826)


Set up .env file:

Create a .env file in the root directory and add your API keys:


![Screenshot 2024-11-23 171539](https://github.com/user-attachments/assets/00937d2e-86da-4c9f-9333-a444ec0d9bf9)


Run the application:


![Screenshot 2024-11-23 171636](https://github.com/user-attachments/assets/ba45e2d9-3b8d-4703-b8ca-3251ddfdff9c)


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


## Performance Comparision Details

1.Accuracy:
Measures how often the system provides the correct answer to a user's query.

Traditional approaches often fail to understand the context, leading to inaccurate results.
The Multimodal RAG leverages LLMs and vector-based search, ensuring higher accuracy.

2.Relevance:
Reflects the closeness of the retrieved information to the user’s query.

Keyword-based methods depend on literal matches, which can overlook semantic meaning.
Multimodal RAG excels in semantic understanding, providing better-aligned results.

3.Response Time:
Tracks the time taken to process and deliver an answer.

Traditional systems are faster for simple queries but struggle with complex documents.
Multimodal RAG is optimized for efficiency while maintaining high relevance and accuracy.

## Line Graph Visualization:
The graph below represents the performance metrics comparison:

X-axis: Metrics (Accuracy, Relevance, Response Time).
Y-axis: Performance in percentage.
Two lines represent Traditional Search (red) and Multimodal RAG (blue).



![Performance_Comparison_Traditional_vs_RAG](https://github.com/user-attachments/assets/cbdded64-90d2-4d67-9997-cd2fef670644)



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




















