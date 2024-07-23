from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
import io


load_dotenv()


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def get_gemini_response(question, context):
    response = chat.send_message(question + "\nContext: " + context, stream=True)
    return response

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(io.BytesIO(pdf_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


st.set_page_config(page_title="RAG Q&A Demo")

st.header("Gemini RAG LLM Application")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'pdf_context' not in st.session_state:
    st.session_state['pdf_context'] = ""

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    st.session_state['pdf_context'] = extract_text_from_pdf(uploaded_file)
    st.success("PDF file uploaded and text extracted.")

input = st.text_input("Input: ", key="input")
submit = st.button("Ask the question")

if submit and input:
    if st.session_state['pdf_context']:
        response = get_gemini_response(input, st.session_state['pdf_context'])
        st.session_state['chat_history'].append(("You", input))
        st.subheader("The Response is")
        for chunk in response:
            st.write(chunk.text)
            st.session_state['chat_history'].append(("Bot", chunk.text))
    else:
        st.warning("Please upload a PDF file first.")

st.subheader("The Chat History is")
for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")
