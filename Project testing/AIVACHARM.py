import os
os.environ['OPENAI_API_KEY'] = "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"
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
import warnings
warnings.filterwarnings("ignore")
from util_dp import get_answer, text_to_speech, autoplay_audio, speech_to_text
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *
import streamlit as st
import PyPDF2
# Initialize floating features for the interface
float_init()
# Initialize session state for managing chat messages
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! How may I assist you today?"}]

initialize_session_state()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
st.title(" AIVA 🤖 Conversational Chatbot ")

# Create a container for the microphone and audio recording
footer_container = st.container()
with footer_container:
    audio_bytes = audio_recorder()

if audio_bytes:
    with st.spinner("Transcribing..."):
        # Write the audio bytes to a temporary file
        webm_file_path = "temp_audio.mp3"
        with open(webm_file_path, "wb") as f:
            f.write(audio_bytes)

        # Convert the audio to text using the speech_to_text function
        transcript = speech_to_text(webm_file_path)
        if transcript:
            st.session_state.messages.append({"role": "user", "content": transcript})
            with st.chat_message("user"):
                st.write(transcript)
            os.remove(webm_file_path)
'''if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            final_response = get_answer(st.session_state.messages, pdf_content)'''
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking🤔..."):
            final_response = get_answer(st.session_state.messages)
        with st.spinner("Generating audio response..."):
            audio_file = text_to_speech(final_response)
            autoplay_audio(audio_file)
        st.write(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        os.remove(audio_file)

footer_container.float("bottom: 0rem;")

