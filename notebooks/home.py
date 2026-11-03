# UI FOR THE HOME PAGE

import streamlit as st
st.title('Welcome to AI Research Agent for KPMG Clients')
st.write('Upload your research documents below. This chatbot will scan them and provide business-friendly insights.')

# Upload documents
uploaded_files = st.file_uploader("Upload your documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        st.write(f"Uploaded: {file.name}")

# Question box for user to interact with
question = st.text_input("Ask a question about your documents")


