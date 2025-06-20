import os
import dotenv
from time import time
import streamlit as st
from io import BytesIO

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["USER_AGENT"] = "MyFastAPIApp/1.0"
dotenv.load_dotenv()
os.environ["USER_AGENT"] = "myagent"

import openai
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Constants
DEFAULT_MODEL_SIZE = "medium"
DEFAULT_CHUNK_LENGTH = 10
STORAGE_PATH = 'faiss_index'
openai.api_key = os.getenv("OPENAI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI models with API key from Streamlit secrets
EMBEDDING_MODEL = OpenAIEmbeddings(api_key=openai_api_key)
DOCUMENT_VECTOR_DB = None  # Will be initialized when documents are added
LANGUAGE_MODEL = ChatOpenAI(api_key=openai_api_key)

PROMPT_TEMPLATE = """
You are chatbot that answer to Current Query. Always use the context to answer the Current Query

Incase context is blank or not provided, reply back with "Please upload a document first to create the knowledge base for Q&A"
incase if the context cannot answer the Current Query properly, then reply back "I dont have the context to answer this question"

Previous Conversation:{chat_history}
Current Query: {user_query} 
context: {document_context} 
Answer:
"""

class ConversationHistory:
    def __init__(self, max_history=5):
        self.history = []
        self.max_history = max_history
    
    def add_exchange(self, user_input, assistant_response):
        self.history.append({
            "user": user_input,
            "assistant": assistant_response
        })
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_formatted_history(self):
        formatted = ""
        for exchange in self.history:
            formatted += f"Customer: {exchange['user']}\n"
            formatted += f"Assistant: {exchange['assistant']}\n"
        return formatted

def load_pdf_documents(uploaded_files=None):

    documents = []
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Convert uploaded file to BytesIO
            pdf_file = BytesIO(uploaded_file.getvalue())
            # Create a temporary file to store the PDF
            with open(f"temp_{uploaded_file.name}", "wb") as f:
                f.write(pdf_file.getvalue())
            # Load the document
            document_loader = PyPDFLoader(f"temp_{uploaded_file.name}")
            documents.extend(document_loader.load())
            # Clean up temporary file
            os.remove(f"temp_{uploaded_file.name}")
    else:
        # Fallback to default path if no files uploaded
        documents = "No pdfs uploaded"
    
    return documents

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    global DOCUMENT_VECTOR_DB
    if DOCUMENT_VECTOR_DB is None:
        DOCUMENT_VECTOR_DB = FAISS.from_documents(document_chunks, EMBEDDING_MODEL)
    else:
        DOCUMENT_VECTOR_DB.add_documents(document_chunks)
    
    # Save the FAISS index
    if not os.path.exists(STORAGE_PATH):
        os.makedirs(STORAGE_PATH)
    DOCUMENT_VECTOR_DB.save_local(STORAGE_PATH)

def load_existing_index():
    global DOCUMENT_VECTOR_DB
    if os.path.exists(STORAGE_PATH):
        DOCUMENT_VECTOR_DB = FAISS.load_local(STORAGE_PATH, EMBEDDING_MODEL)
        return True
    return False

def find_related_documents(query):
    if DOCUMENT_VECTOR_DB is None:
        return []
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents, chat_history):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL

    print('context:',context_documents)
    return response_chain.invoke({
        "user_query": user_query, 
        "document_context": context_documents,
        "chat_history": chat_history
    })


