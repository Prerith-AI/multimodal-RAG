
# Standard Library Import
import os
from dotenv import load_dotenv
import cv2
import pytesseract
from typing import List,Dict

# Data Processing and Machine Learning Libraries
import numpy as np
import pandas as pd

# LangChain Imports for Document Processing
from langchain.docstore.document import Document
from langchain.document_loaders import (
    PDFLoader, 
    TextLoader, 
    UnstructuredMarkdownLoader,
    UnstructuredCSVLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

# LangChain AI Model Imports
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Web Framework and UI
import streamlit as st
load_dotenv()
class MultimodalDocumentLoader:
    """
    Advanced document loader supporting multiple file formats
    
    Capabilities:
    - Extract text from PDF, Text, Markdown, CSV, Excel
    - OCR for image-based documents
    - Intelligent text chunking
    """
    def __init__(self):
        # Configure Tesseract OCR for image text extraction
        pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
        
        # Initialize intelligent text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,    # Optimal chunk size for semantic coherence
            chunk_overlap=200   # Ensures contextual continuity
        )

    def load_image(self, image_path: str) -> List[Document]:
        """
        Extract and process text from images using advanced OCR
        
        Processing Strategy:
        1. Read image with OpenCV
        2. Extract text using Tesseract
        3. Split text into semantic chunks
        4. Create LangChain Documents
        """
        image = cv2.imread(image_path)
        text = pytesseract.image_to_string(image)
        
        # Intelligent text chunking
        text_chunks = self.text_splitter.split_text(text)
        
        return [
            Document(
                page_content=chunk, 
                metadata={'source': image_path}
            ) for chunk in text_chunks
        ]

    def load_document(self, file_path: str) -> List[Document]:
        """
        Dynamically process documents based on file type
        
        Supported Types:
        - PDF
        - Text
        - Markdown
        - CSV
        - Excel
        - Images (PNG, JPG, JPEG)
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Loader mapping for different document types
        loaders = {
            '.pdf': PDFLoader,
            '.txt': TextLoader,
            '.md': UnstructuredMarkdownLoader,
            '.csv': UnstructuredCSVLoader,
            '.xlsx': UnstructuredExcelLoader
        }
        
        # Special handling for image files
        if file_extension in ['.png', '.jpg', '.jpeg', '.bmp']:
            return self.load_image(file_path)
        
        # Process document using appropriate loader
        if file_extension in loaders:
            loader = loaders[file_extension](file_path)
            documents = loader.load()
            return self.text_splitter.split_documents(documents)
        
        raise ValueError(f"Unsupported file type: {file_extension}")

class RAGEmbedder:
    """
    Vector embedding and semantic search engine
    
    Key Features:
    - Convert documents to semantic vector representations
    - Efficient similarity search
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Initialize sentence transformer for embedding
        self.embedding_function = SentenceTransformerEmbeddings(
            model_name=model_name
        )
        self.vectorstore = None

    def embed_documents(self, documents: List[Document]):
        """
        Create vector representation of documents
        
        Process:
        1. Convert documents to vector embeddings
        2. Create FAISS index for efficient search
        """
        self.vectorstore = FAISS.from_documents(
            documents, 
            self.embedding_function
        )
        return self.vectorstore

    def similarity_search(self, query: str, k: int = 3):
        """
        Retrieve most semantically relevant documents
        
        Strategy:
        - Embed query
        - Find top-k most similar document chunks
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        return self.vectorstore.similarity_search(query, k=k)

class RAGLanguageModel:
    """
    Intelligent Language Model Configuration
    
    Features:
    - Text processing with Claude
    - Image understanding with GPT-4o Mini
    """
    def __init__(
        self, 
        text_model='claude-3-haiku-20240307', 
        image_model='gpt-4o-mini'
    ):
        # Initialize text processing model (Claude)
        self.text_llm = ChatAnthropic(
            model=text_model, 
            anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')
        )
        
        # Initialize image processing model (GPT-4o Mini)
        self.image_llm = ChatOpenAI(
            model=image_model,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )

    def generate_rag_chain(self):
        """
        Create Retrieval-Augmented Generation processing pipeline
        
        Components:
        - Context-aware prompt template
        - Language model integration
        - Response parsing
        """
        # Structured prompt for contextual Q&A
        prompt = ChatPromptTemplate.from_template("""
        You are an expert AI assistant. Provide a precise answer based on the given context.
        
        Context: {context}
        Question: {question}
        
        Detailed Answer:""")
        
        # Construct RAG processing chain
        rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt 
            | self.text_llm
            | StrOutputParser()
        )
        
        return rag_chain

def main():
   
    st.set_page_config(
        page_title="Multimodal RAG Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title('Advanced Multimodal RAG System')
    
    # Secure API key management
    os.environ['ANTHROPIC_API_KEY'] = st.secrets.get('ANTHROPIC_API_KEY')
    os.environ['OPENAI_API_KEY'] = st.secrets.get('OPENAI_API_KEY')
    
    # Initialize RAG components
    document_loader = MultimodalDocumentLoader()
    rag_embedder = RAGEmbedder()
    rag_model = RAGLanguageModel()
    
    # Document upload interface
    st.sidebar.header("Document Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Choose documents", 
        accept_multiple_files=True,
        type=['pdf', 'txt', 'md', 'csv', 'xlsx', 'png', 'jpg', 'jpeg']
    )
    
    # Main content area for Q&A
    st.header("Ask Questions About Your Documents")
    
    # Process uploaded documents
    if uploaded_files:
        with st.spinner('Processing documents...'):
            all_documents = []
            for file in uploaded_files:
                # Temporary file saving
                with open(file.name, 'wb') as f:
                    f.write(file.getbuffer())
                
                # Load and chunk documents
                documents = document_loader.load_document(file.name)
                all_documents.extend(documents)
            
            # Create vector store
            vectorstore = rag_embedder.embed_documents(all_documents)
            
            # Question input
            question = st.text_input("Enter your question")
            
            if question:
                with st.spinner('Generating answer...'):
                    # Retrieve relevant documents
                    retrieved_docs = rag_embedder.similarity_search(question)
                    
                    # Create RAG chain
                    rag_chain = rag_model.generate_rag_chain()
                    
                    # Combine retrieved documents
                    context = "\n".join([doc.page_content for doc in retrieved_docs])
                    
                    # Generate answer
                    answer = rag_chain.invoke({
                        "context": context, 
                        "question": question
                    })
                    
                    # Display results
                    st.subheader("AI Assistant's Response")
                    st.write(answer)
                    
                    # Show retrieved document snippets
                    st.subheader("Retrieved Document Snippets")
                    for doc in retrieved_docs:
                        st.text(doc.page_content[:500] + "...")

if __name__ == "__main__":
    main()

