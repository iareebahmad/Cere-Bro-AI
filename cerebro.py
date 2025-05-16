import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# Set page config
st.set_page_config(page_title="Cere-Bro AI")

# Get user input
st.markdown(
    """
    <h2 style='font-size:28px; font-weight:bold;'>
    Bro, whassup? No cap, Cere-Bro is here to make your life way easier.
    </h2>
    """,
    unsafe_allow_html=True
)

user_prompt = st.text_input("Ask me anything.")

# Secure API key from secrets
OPENAI_API_KEY = st.secrets["openai"]["api_key"]

# Header and file upload
st.header("Cere-Bro AI")

with st.sidebar:
    st.image("logo.png", width=150)
    st.title("Your Documents")
    file = st.file_uploader("Upload your files to Cere-Bro AI and ask your questions", type="pdf")

# Process the uploaded PDF
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=20,
        chunk_overlap=10,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Create FAISS vector store
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    # If user has entered a prompt
    if user_prompt:
        match = vector_store.similarity_search(user_prompt)

        # Define language model
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0.1,
            max_tokens=500,
            model_name="gpt-4o-mini"
        )

        # Create and run QA chain
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_prompt)
        st.write(response)
