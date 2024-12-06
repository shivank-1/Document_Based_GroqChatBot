import streamlit as st
import os
from langchain_groq import ChatGroq
#from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import pickle
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
def download_embeddings():
    embedding_path = "local_embeddings"

    if os.path.exists(embedding_path):
        with open(embedding_path, 'rb') as f:
            embedding = pickle.load(f)
    else:
        embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        with open(embedding_path, 'wb') as f:
            pickle.dump(embedding, f)
    return embedding
groq_api_key=os.getenv("groqkey")
llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.1-70b-versatile",temperature=0.82)
prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate respone based on the question
    <context>
    {context}
    <context>
    Question:{input}

    """

)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=download_embeddings()
        st.session_state.loader=PyPDFDirectoryLoader("Research_Papers") ## Data Ingestion step
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

st.title("RAG Document Q&A With Groq And Lama3")

user_prompt=st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")

import time

if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt) #it create a chain for passing list of document to the model
    retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 5})
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()
    response=retrieval_chain.invoke({'input':user_prompt})
    print(f"Response time :{time.process_time()-start}")

    st.write(response['answer'])

    ## With a streamlit expander
    with st.expander("Document similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')



