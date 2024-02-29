import os
import speech_recognition as sr
from gtts import gTTS
import pygame
from langdetect import detect

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = "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"

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak:")
        audio = recognizer.listen(source)
        print("Processing...")

    try:
        # Recognize the spoken text using language detection
        detected_language = detect(recognizer.recognize_google(audio))
        print(f"Detected Language: {detected_language}")
        
        # Ask the user for the desired language
        print("Choose the desired language (en for English, hi for Hindi, mr for Marathi):")
        desired_language = input().lower()

        # Recognize the spoken text using the specified or detected language
        text = recognizer.recognize_google(audio, language=f"{desired_language}-{desired_language.upper()}")
        return text, detected_language
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def text_to_speech(text, lang="mr"):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")

    pygame.mixer.init()
    pygame.mixer.music.load("output.mp3")
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

if __name__ == '__main__':
    print("Langchain start...")

    # Use STT to convert spoken words into text
    user_input, detected_language = speech_to_text()
    if user_input and detected_language:
        print(f"User Input ({detected_language}):", user_input)

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
        text_to_speech(result, detected_language)
