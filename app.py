import streamlit as st
import os
import dotenv
import uuid

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

from rag_methods import (
    load_pdf_documents,
    chunk_documents,
    index_documents,
    find_related_documents,
    ConversationHistory,
    generate_answer
)

os.environ["USER_AGENT"] = "MyFastAPIApp/1.0"
dotenv.load_dotenv()

st.set_page_config(
    page_title="RAG Chatbot", 
    page_icon="🤖", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

# --- Header ---
st.title("RAG Chatbot")

# --- Initial Setup ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Upload a PDF document and ask me questions about it!"}
    ]

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = ConversationHistory()

# --- Side Bar ---
with st.sidebar:
    st.header("Document Upload")
    
    # File upload input for RAG with documents
    uploaded_files = st.file_uploader(
        "📄 Upload a PDF document", 
        type=["pdf"],
        accept_multiple_files=True,
        key="rag_docs",
    )

    # Process uploaded documents
    if uploaded_files:
        # Load and process documents
        documents = load_pdf_documents(uploaded_files)
        chunks = chunk_documents(documents)
        index_documents(chunks)
        
        # Update sources list
        st.session_state.rag_sources = [file.name for file in uploaded_files]

    # Show uploaded documents
    if st.session_state.rag_sources:
        st.subheader("Uploaded Documents:")
        for source in st.session_state.rag_sources:
            st.write(f"- {source}")

    st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")

# --- Main Chat Interface ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your document"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process the query
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Find relevant documents
        context = find_related_documents(prompt)
        
        # Get chat history
        chat_history = st.session_state.conversation_history.get_formatted_history()
        
        # Generate response
        response = generate_answer(prompt, context, chat_history)
        
        # Update conversation history
        st.session_state.conversation_history.add_exchange(prompt, response.content)
        
        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": response.content})
        st.markdown(response.content)



    
