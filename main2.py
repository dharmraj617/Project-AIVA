import os
import speech_recognition as sr
from gtts import gTTS
import pygame

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI

import warnings
warnings.filterwarnings("ignore")

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = "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"

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak:")
        audio = recognizer.listen(source)
        print("Processing...")
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None

def text_to_speech(text):
    tts = gTTS(text=text, lang="en")
    tts.save("output.mp3")

    pygame.mixer.init()
    pygame.mixer.music.load("output.mp3")
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

if __name__ == '__main__':
    print("Langchain start...")

    # Use STT to convert spoken words into text
    user_input = speech_to_text()
    if user_input:
        print("User Input:", user_input)

        # Continue with the rest of your code...

        pdf_path = "DYPCET.pdf"
        loader = PyPDFLoader(file_path=pdf_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator='\n')
        docs = text_splitter.split_documents(documents=documents)

        embedding = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embedding)
        vectorstore.save_local("faiss_index_react")

        new_vectorstore = FAISS.load_local("faiss_index_react", embedding)

        qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever())

        result = qa.run(user_input)
        print("Answer:", result)

        # Use TTS to convert the answer to speech and play
        text_to_speech(result)
