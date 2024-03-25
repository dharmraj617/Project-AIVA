import os
os.environ['OPENAI_API_KEY'] = "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"
from openai import OpenAI
import base64
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
import warnings
warnings.filterwarnings("ignore")

client = OpenAI(api_key= "OPENAI_API_KEY" )

def get_answer(messages):
    USER_INPUT = audio_data
    pdf_path = "DYPCET v3.0.pdf"

    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator='\n')
    docs = text_splitter.split_documents(documents=documents)

    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embedding)
    vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local("faiss_index_react", embedding)

    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever())

    result = qa.run(USER_INPUT)
    response = result
    return response

def speech_to_text(audio_data):
    with open(audio_data, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file
        )
    return transcript

def text_to_speech(input_text):
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=input_text
    )
    webm_file_path = "temp_audio_play.mp3"
    with open(webm_file_path, "wb") as f:
        response.stream_to_file(webm_file_path)
    return webm_file_path

def autoplay_audio(file_path:str):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 =base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src = "data:audio/mp3;base64,{b64}" type = "audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html = True)
