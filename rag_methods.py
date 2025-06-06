import os
import dotenv
from time import time
import streamlit as st
import logging

from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    WebBaseLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
)
# pip install docx2txt, pypdf
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

os.environ["USER_AGENT"] = "MyFastAPIApp/1.0"
dotenv.load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DB_DOCS_LIMIT = 10

# Function to stream the response of the LLM 
def stream_llm_response(llm_stream, messages):
    response_message = ""

    for chunk in llm_stream.stream(messages):

        response_message += chunk.content
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})


# --- Indexing Phase ---

def load_doc_to_db():
    # Use loader according to doc type
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = [] 
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    os.makedirs("source_files", exist_ok=True)
                    file_path = f"./source_files/{doc_file.name}"
                    with open(file_path, "wb") as file:
                        file.write(doc_file.read())

                    try:
                        if doc_file.type == "application/pdf":
                            loader = PyPDFLoader(file_path)
                        elif doc_file.name.endswith(".docx"):
                            loader = Docx2txtLoader(file_path)
                        elif doc_file.type in ["text/plain", "text/markdown"]:
                            loader = TextLoader(file_path)
                        else:
                            st.warning(f"Document type {doc_file.type} not supported.")
                            continue

                        docs.extend(loader.load())
                        st.session_state.rag_sources.append(doc_file.name)

                    except Exception as e:
                        st.toast(f"Error loading document {doc_file.name}: {e}", icon="⚠️")
                        print(f"Error loading document {doc_file.name}: {e}")
                    
                    finally:
                        os.remove(file_path)

                else:
                    st.error(F"Maximum number of documents reached ({DB_DOCS_LIMIT}).")

        if docs:
            _split_and_load_docs(docs)
            st.toast(f"Document *{str([doc_file.name for doc_file in st.session_state.rag_docs])[1:-1]}* loaded successfully.", icon="✅")


def load_url_to_db():
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        docs = []
        if url not in st.session_state.rag_sources:
            if len(st.session_state.rag_sources) < 10:
                try:
                    loader = WebBaseLoader(url)
                    docs.extend(loader.load())
                    st.session_state.rag_sources.append(url)

                except Exception as e:
                    st.error(f"Error loading document from {url}: {e}")

                if docs:
                    _split_and_load_docs(docs)
                    st.toast(f"Document from URL *{url}* loaded successfully.", icon="✅")

            else:
                st.error("Maximum number of documents reached (10).")



def initialize_vector_db(docs):
    if "AZ_OPENAI_API_KEY" not in os.environ:
        embedding = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
    else:
        embedding = AzureOpenAIEmbeddings(
            api_key=os.getenv("AZ_OPENAI_API_KEY"), 
            azure_endpoint=os.getenv("AZ_OPENAI_ENDPOINT"),
            model="text-embedding-3-large",
            openai_api_version="2024-02-15-preview",
        )

    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name=f"{str(time()).replace('.', '')[:14]}_" + st.session_state['session_id'],
    )


    # We need to manage the number of collections that we have in memory, we will keep the last 20
    chroma_client = vector_db._client
    collection_names = sorted([collection.name for collection in chroma_client.list_collections()])
    print("Number of collections:", len(collection_names))
    while len(collection_names) > 200:
        chroma_client.delete_collection(collection_names[0])
        collection_names.pop(0)

    return vector_db


def _split_and_load_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    document_chunks = text_splitter.split_documents(docs)

    if "vector_db" not in st.session_state:
        st.session_state.vector_db = initialize_vector_db(document_chunks)
    else:
        st.session_state.vector_db.add_documents(document_chunks)


# --- Retrieval Augmented Generation (RAG) Phase ---



def get_conversational_rag_chain(llm,messages):

    retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 2})
    user_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", """ The input given above maybe of 2 scenarios:
         1. Incase if user has asked to quiz him, then provide the question as per the topic / context given
         2. Incase user has given an answer to a question in the form of an option for an MCQ question,  mention if the option is correct or not for the recent question. 
         """),
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, user_prompt)


        ################# logging purpose only ##################
    logging.info(f"☑ ✓ ✅Logs retrieved for message:{messages[-1]}")
    retrieval_output = retriever.get_relevant_documents(f"{messages[-1]} Given the option/input chosen by the user, mention if the option is correct or not. Inboth cases, give an explaination focussing on most recent question. if user asked to quiz him, then provide the question as per context")
    
    if retrieval_output:
        logging.info(f"++Logs++Retrieved {len(retrieval_output)} documents.")

        for i, doc in enumerate(retrieval_output):
            content_length = len(doc.page_content)
            logging.info(f"📄 Doc {i+1} length: {content_length} characters")
            logging.info(f"Content :{doc.page_content}")
 

    #####################################################################    
    system_prompt = ChatPromptTemplate.from_messages([
        ("system",
        """You are a quiz master that ask questions to user. you will ask user a question and give 4 options. only one opion will be correct.make sure all 4 options are shown in 4 lines
        You will have some context to help with your asking the questions and deciding the correct option, but now always would be completely related or helpful.
        Only use the context provided to provide response. do not hallucinate. if you dont have the context, just say so.
        Incase if the user has provided a right or wrong anser,ask him if which topic he would like to get quizzed on\n
        {context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, system_prompt)

    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def stream_llm_rag_response(llm_stream, messages):
    conversation_rag_chain = get_conversational_rag_chain(llm_stream,messages)
    response_message = "*(RAG Response)*\n"
    for chunk in conversation_rag_chain.pick("answer").stream({"messages": messages[:-1], "input": messages[-1].content}):
        response_message += chunk
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})
