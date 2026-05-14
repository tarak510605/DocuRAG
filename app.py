import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import pathlib

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
API_KEY_SECRET = SecretStr(API_KEY) if API_KEY else None

CHAT_MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", api_key=API_KEY_SECRET)
    FAISS_INDEX_PATH = pathlib.Path("faiss_index")
    
    # Create a new FAISS index from text chunks
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local(str(FAISS_INDEX_PATH))
    st.success("Vector store created and saved.")

def get_conversational_chain(model_name):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, just say "answer is not available in the context", 
    don't provide the wrong answer.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model=model_name, temperature=0.3, api_key=API_KEY_SECRET)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", api_key=API_KEY_SECRET)
    FAISS_INDEX_PATH = pathlib.Path("faiss_index")

    # Check if FAISS index exists
    if not (FAISS_INDEX_PATH / "index.faiss").exists():
        st.error("No FAISS index found. Please upload and process PDFs first.")
        return

    new_db = FAISS.load_local(str(FAISS_INDEX_PATH), embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    last_error = None
    for idx, model_name in enumerate(CHAT_MODEL_CANDIDATES):
        try:
            chain = get_conversational_chain(model_name)
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            if idx > 0:
                st.info(f"Primary model quota hit; answered with fallback model: {model_name}")
            st.write("Reply:", response["output_text"])
            return
        except Exception as exc:
            last_error = exc
            message = str(exc)
            if "RESOURCE_EXHAUSTED" not in message and "429" not in message:
                st.error(f"Error while generating answer: {message}")
                return

    st.error("All available chat models are currently rate-limited for this API key. Please wait and retry, or use a key with higher quota.")
    if last_error:
        st.caption(str(last_error))

def main():
    st.set_page_config("DocuRAG")
    st.header("DocuRAG: Chat with Multiple PDF using Gemini")

    if not API_KEY:
        st.error("GOOGLE_API_KEY not found. Please add it to the .env file and restart Streamlit.")
        return

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
