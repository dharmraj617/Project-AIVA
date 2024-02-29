import os
os.environ['OPENAI_API_KEY'] = "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"
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

if __name__ == '__main__':
    print("Langchain start...")
    pdf_path = "Python Cheet Sheet.pdf"

    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator='\n')
    docs = text_splitter.split_documents(documents=documents)

    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embedding)
    vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local("faiss_index_react", embedding)

    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever())

    result = qa.run("What are different operators in Python?")
    print(result)
