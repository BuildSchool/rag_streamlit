import os
import re
import streamlit as st
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain.chat_models import init_chat_model
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# Load environment variables
load_dotenv()

# Streamlit page setup
st.set_page_config(page_title="Syed Hasan RAG", layout="wide", page_icon="📄")

# Inject custom CSS for backgrounds, rounded containers, and spacing
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #e0e7ef 100%);
        border-radius: 18px;
        padding: 2rem 2.5rem 2rem 2.5rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 24px 0 rgba(0,0,0,0.07);
    }
    .block-container {
        padding-top: 2.5rem !important;
    }
    .custom-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
        margin-top: 2.5rem;
        padding-top: 1.5rem;
    }
    .custom-header img {
        height: 48px;
        border-radius: 12px;
    }
    .custom-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2d3748;
        letter-spacing: -1px;
    }
    /* Chat input box styling */
    section[data-testid="stChatInput"] textarea, .stTextInput>div>div>input {
        border: 3px solid #6C63FF !important;
        box-shadow: 0 2px 12px 0 rgba(108,99,255,0.13) !important;
        border-radius: 16px !important;
        padding: 1.1rem 1.2rem !important;
        font-size: 1.15rem !important;
        background: #fff !important;
        color: #22223b !important;
    }
    section[data-testid="stChatInput"] textarea::placeholder, .stTextInput>div>div>input::placeholder {
        color: #6C63FF99 !important;
        font-weight: 500 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom header with logo, styled title, and subtitle
st.markdown(
    '<div class="custom-header">'
    '<img src="https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png" alt="Logo">'
    '<div>'
    '<span class="custom-title" style="color:#6C63FF;">Chat with youtube and your PDF</span>'
    '<br>'
    '<span style="font-size:1.1rem;font-weight:500;color:#3b3b3b;background:#e0e7ef;padding:4px 12px;border-radius:8px;">Created by Syed Hasan (First RAG application)</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)

# Highlight chat and input areas
st.markdown(
    '''<style>
    .stChatMessage, .stTextInput, .stTextArea, .stButton, .stFileUploader, .stTextInput>div>div>input {
        background: #fff !important;
        border: 2px solid #6C63FF !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px 0 rgba(108,99,255,0.07);
    }
    .stChatMessage {
        margin-bottom: 1.2rem !important;
    }
    .stTextInput>div>div>input {
        font-size: 1.1rem !important;
        color: #22223b !important;
    }
    .stTextInput label, .stTextArea label {
        color: #6C63FF !important;
        font-weight: 600 !important;
    }
    .stButton>button {
        background: linear-gradient(90deg,#6C63FF 60%,#48b1f3 100%) !important;
        color: #fff !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        border: none !important;
    }
    .stFileUploader>div>div {
        border: 2px dashed #6C63FF !important;
        background: #f8fafc !important;
    }
    </style>''',
    unsafe_allow_html=True
)

# Sidebar instructions
st.sidebar.markdown(
    """
    <div style='background:#e0e7ef;padding:18px 14px 14px 14px;border-radius:12px;margin-bottom:18px;'>
    <h4 style='color:#6C63FF;margin-bottom:0.5rem;'>How to use</h4>
    <ul style='font-size:1rem;color:#22223b;padding-left:1.2em;'>
      <li>Paste a YouTube video or Shorts URL in the box above.</li>
      <li>Or upload a PDF file (max 200MB).</li>
      <li>Ask any question about the content in the chat below.</li>
      <li>Both YouTube and PDF can be used together for richer answers.</li>
      <li>All processing is local and secure.</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# Validate YouTube URL
def is_valid_youtube_url(url: str) -> bool:
    if not url:
        return False
    youtube_regex = r'(https?://)?(www\.)?(youtube\.com/(watch\?v=|shorts/)|youtu\.be/)[a-zA-Z0-9_-]{11}$'
    return bool(re.match(youtube_regex, url))

# Initialize chat history and vector DB if not present
if "history" not in st.session_state:
    st.session_state.history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# Sidebar Inputs
st.sidebar.header("📩 Source Input")
youtube_url = st.sidebar.text_input("📹 YouTube URL")
pdf_file = st.sidebar.file_uploader("📄 Upload PDF", type=["pdf"])

# If a new PDF is uploaded, process and cache it
if pdf_file is not None and (st.session_state.doc_name != pdf_file.name):
    pdf_path = os.path.join("./", pdf_file.name)
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.getbuffer())
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    splits = text_splitter.split_documents(docs)
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    db = FAISS.from_documents(splits, embedding)
    st.session_state.vector_db = db
    st.session_state.doc_name = pdf_file.name

# Render previous chat history
for user_msg, bot_msg in st.session_state.history:
    st.chat_message("user", avatar="🧑").markdown(user_msg)
    st.chat_message("assistant", avatar="🤖").markdown(bot_msg)

# Chat input
user_input = st.chat_input("Ask a question...")

if user_input:
    if not (st.session_state.vector_db or (youtube_url and is_valid_youtube_url(youtube_url))):
        st.error("\u274c Please provide at least one source!!!")
    else:
        with st.spinner("Generating answer \u23f3"):
            loaders = []
            # Use cached PDF vector DB if available
            db = st.session_state.vector_db
            # Handle YouTube URL if provided
            if youtube_url:
                if is_valid_youtube_url(youtube_url):
                    try:
                        loaders.append(
                            YoutubeLoader.from_youtube_url(youtube_url, add_video_info=False)
                        )
                    except Exception as e:
                        st.error(f"\u274c Could not process YouTube video: {str(e)}")
                        loaders = []
                else:
                    st.error("\u274c Invalid YouTube URL")
                    st.stop()
            # If YouTube is used, process and merge with PDF DB
            if loaders:
                docs = []
                for loader in loaders:
                    try:
                        docs.extend(loader.load())
                    except Exception as e:
                        st.error(f"\u274c Could not fetch transcript for the YouTube video. It may not be available for Shorts or this video.\nError: {str(e)}")
                if not docs:
                    st.info("No transcript could be loaded from the provided YouTube link. Only PDF content will be used.")
                else:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=150,
                        separators=["\n\n", "\n", ".", " "]
                    )
                    splits = text_splitter.split_documents(docs)
                    embedding = OpenAIEmbeddings(
                        model="text-embedding-3-large",
                        api_key=os.getenv("OPENAI_API_KEY")
                    )
                    db_youtube = FAISS.from_documents(splits, embedding)
                    # Merge YouTube and PDF DBs if both exist
                    if db:
                        db.merge_from(db_youtube)
                    else:
                        db = db_youtube
            # Initialize LLM
            llm = init_chat_model(
                "gpt-4.1-mini",
                model_provider="openai",
                api_key=os.getenv("OPENAI_API_KEY")
            )
            # Ensure db is not None before proceeding
            if db is None:
                st.error("No valid content found in the provided sources. Please upload a PDF or use a YouTube video with a transcript.")
                st.stop()
            system_text = (
                "You are a context-aware assistant.\n"
                "Use the provided documents to answer user queries.\n"
                "If the answer is not in the docs, say you don't know."
            )
            condensed_template = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_text),
                HumanMessagePromptTemplate.from_template(
                    "Given the conversation and question, rephrase the question if needed for clarity.\n"
                    "Conversation: {chat_history}\nQuestion: {question}"
                )
            ])
            qa_template = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_text),
                HumanMessagePromptTemplate.from_template(
                    "Context: {context}\nConversation: {chat_history}\nQuestion: {question}"
                )
            ])
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm,
                retriever=db.as_retriever(),
                condense_question_prompt=condensed_template,
                combine_docs_chain_kwargs={"prompt": qa_template},
                return_source_documents=False
            )
            result = qa_chain({
                "question": user_input,
                "chat_history": st.session_state.history
            })
            answer = result.get("answer")
            st.session_state.history.append((user_input, answer))
            st.chat_message("user", avatar="🧑").markdown(user_input)
            st.chat_message("assistant", avatar="🤖").markdown(answer)
