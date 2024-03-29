#import os
#os.environ['OPENAI_API_KEY'] = ""
import streamlit as st
import dotenv
from langchain_openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os

from output_parser import ChatbotResult, chatbot_result

dotenv.load_dotenv()

def langchain_result(uploaded_file, query_text: str) -> str:
    if uploaded_file is not None:
        documents = uploaded_file.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator='\n')
        docs = text_splitter.split_documents(documents=documents)
        embedding = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embedding)
        vectorstore.save_local("faiss_index_react")
        new_vectorstore = FAISS.load_local("faiss_index_react", embedding, allow_dangerous_deserialization=True)

        qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever())
        return qa.invoke(query_text)



st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

pdf_path = "DYPCET v3.0.pdf"
uploaded_file = PyPDFLoader(file_path=pdf_path)
query_text = st.text_input('Enter your question:',
                           placeholder = 'Example: Engineering courses in this college?',
                           disabled=not uploaded_file)
result = []

with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit', disabled=not (uploaded_file and query_text))
    if submitted:
        with st.spinner('Calculating...'):
            response = langchain_result(uploaded_file, query_text)

if len(result):
    st.info(response)