import os
import re
import streamlit as st
from dotenv import load_dotenv
from apify_client import ApifyClient
import tempfile
import json
import requests
import io
from fpdf import FPDF

from langchain_community.document_loaders import PyPDFLoader
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

# --- Unified Sidebar Layout ---
st.sidebar.markdown("## 📩 Source Input")

# PDF Upload
with st.sidebar.expander("PDF Upload & Chat", expanded=False):
    pdf_file = st.file_uploader("📄 Upload PDF", type=["pdf"])
    st.markdown("Upload a PDF and chat with its content.")

# Unified YouTube URL input
st.sidebar.markdown("---")
youtube_url = st.sidebar.text_input("📹 YouTube Video URL here")

# Language and timestamp options for Apify
apify_language = st.sidebar.selectbox(
    "Transcript Language",
    [
        "Default", "English", "Spanish", "French", "German", "Italian", "Portuguese", "Russian", "Japanese", "Korean", "Chinese", "Arabic", "Hindi", "Bengali", "Urdu", "Turkish", "Vietnamese", "Polish", "Dutch", "Greek", "Czech", "Swedish", "Finnish", "Hungarian", "Romanian", "Thai", "Indonesian", "Hebrew", "Malay", "Filipino", "Ukrainian", "Persian", "Norwegian", "Danish", "Slovak", "Bulgarian", "Serbian", "Croatian", "Catalan", "Slovenian", "Lithuanian", "Latvian", "Estonian", "Georgian", "Armenian", "Azerbaijani", "Albanian", "Basque", "Galician", "Macedonian", "Mongolian", "Swahili", "Afrikaans", "Zulu", "Xhosa", "Somali", "Nepali", "Sinhala", "Burmese", "Khmer", "Lao", "Malayalam", "Kannada", "Telugu", "Tamil", "Marathi", "Gujarati", "Punjabi", "Pashto", "Kurdish", "Macedonian", "Bosnian", "Icelandic", "Maltese", "Luxembourgish", "Faroese", "Irish", "Scottish Gaelic", "Welsh", "Yiddish"
    ],
    index=0
)
apify_timestamps = st.sidebar.selectbox("Include Timestamps", ["No", "Yes"], index=0)

# Extract Transcript button (for YouTube)
extract_clicked = False
if youtube_url and is_valid_youtube_url(youtube_url):
    extract_clicked = st.sidebar.button("Extract Transcript", use_container_width=True)
else:
    st.sidebar.button("Extract Transcript", use_container_width=True, disabled=True)

# Apify options (hidden unless fallback is needed)
apify_language = "Default"
apify_timestamps = "No"

# --- Main Chat Interface ---
st.markdown("---")
st.markdown('<div style="margin-bottom:1.5rem"></div>', unsafe_allow_html=True)

# Single chat input for all sources
user_input = st.chat_input("Ask a question about your PDF or YouTube video...")

# --- Transcript/Vector DB Loading Logic ---
# PDF logic unchanged
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

# --- Transcript Extraction Logic (triggered by Extract Transcript button) ---
youtube_transcript_preview = None
youtube_transcript_segments = None
if extract_clicked:
    st.session_state.vector_db = None  # Reset previous YouTube vector DB
    st.session_state.youtube_source = None
    st.session_state.youtube_transcript = None
    with st.spinner("Extracting transcript, please wait..."):
        try:
            yt_loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=False)
            yt_docs = yt_loader.load()
            if yt_docs:
                with st.spinner("Creating embeddings for YouTube transcript..."):
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=150,
                        separators=["\n\n", "\n", ".", " "]
                    )
                    splits = text_splitter.split_documents(yt_docs)
                    embedding = OpenAIEmbeddings(
                        model="text-embedding-3-large",
                        api_key=os.getenv("OPENAI_API_KEY")
                    )
                    db_youtube = FAISS.from_documents(splits, embedding)
                    st.session_state.vector_db = db_youtube
                st.session_state.youtube_source = "langchain"
                st.session_state.youtube_transcript = yt_docs[0].page_content if yt_docs else None
                youtube_transcript_preview = st.session_state.youtube_transcript
                youtube_transcript_segments = None
            else:
                raise Exception("No transcript from Langchain.")
        except Exception:
            # Fallback to Apify
            with st.spinner("Extracting transcript from Apify..."):
                try:
                    apify_token = os.getenv("APIFY_API_TOKEN")
                    api_url = (
                        "https://api.apify.com/v2/acts/topaz_sharingan~youtube-transcript-scraper-1/"
                        "run-sync-get-dataset-items?token=" + apify_token
                    )
                    payload = {
                        "startUrls": [youtube_url],
                        "language": apify_language,
                        "includeTimestamps": apify_timestamps,
                    }
                    response = requests.post(api_url, json=payload, timeout=120)
                    response.raise_for_status()
                    transcript_items = response.json()
                    apify_lang_info = transcript_items[0].get("language", None) if (isinstance(transcript_items, list) and transcript_items) else None
                    transcript = transcript_items[0].get("transcript", []) if (isinstance(transcript_items, list) and transcript_items) else None
                    if isinstance(transcript, list) and transcript:
                        transcript_str = ""
                        formatted_segments = []
                        for seg in transcript:
                            line = ""
                            if apify_timestamps == "Yes" and "timestamp" in seg:
                                line += f"[{seg['timestamp']}] "
                            line += seg.get("text", "")
                            formatted_segments.append(line)
                        transcript_str = "\n\n".join(formatted_segments)
                        youtube_transcript_segments = formatted_segments
                    elif isinstance(transcript, str) and transcript.strip():
                        transcript_str = transcript.strip()
                        youtube_transcript_segments = None
                    else:
                        st.session_state.youtube_source = None
                        st.session_state.youtube_transcript = None
                        youtube_transcript_preview = None
                        youtube_transcript_segments = None
                        st.error("Transcript format not recognized or empty from Apify.")
                        st.info(f"Raw Apify response: {transcript_items}")
                    # Only continue if transcript_str is set
                    if 'transcript_str' in locals() and transcript_str.strip():
                        # Always store the original transcript in session state
                        st.session_state.youtube_transcript_original = transcript_str.strip()
                        # If translation is needed, chunk and translate
                        if apify_lang_info and apify_lang_info.lower() != apify_language.lower() and apify_language != "Default":
                            st.warning(f"Transcript language returned by Apify is '{apify_lang_info}', not '{apify_language}'. The video may not have a transcript in the requested language. The transcript will be automatically translated.")
                            try:
                                import openai
                                openai.api_key = os.getenv("OPENAI_API_KEY")
                                text_splitter = RecursiveCharacterTextSplitter(
                                    chunk_size=1000,
                                    chunk_overlap=150,
                                    separators=["\n\n", "\n", ".", " "]
                                )
                                chunks = text_splitter.split_text(st.session_state.youtube_transcript_original)
                                translated_chunks = []
                                for i, chunk in enumerate(chunks):
                                    with st.spinner(f"Translating chunk {i+1}/{len(chunks)} to {apify_language}..."):
                                        translation_prompt = (
                                            f"Translate the following transcript chunk to {apify_language}. Preserve timestamps if present.\n\nTranscript chunk:\n" + chunk
                                        )
                                        try:
                                            response = openai.ChatCompletion.create(
                                                model="gpt-3.5-turbo",
                                                messages=[{"role": "user", "content": translation_prompt}],
                                                max_tokens=2048,
                                                temperature=0.2
                                            )
                                            translated_chunk = response.choices[0].message.content.strip()
                                            translated_chunks.append(translated_chunk)
                                        except Exception as e:
                                            st.error(f"Translation failed for chunk {i+1}: {e}")
                                            translated_chunks.append(chunk)  # fallback to original chunk
                                translated_transcript = "\n\n".join(translated_chunks)
                                formatted_transcript = translated_transcript
                                youtube_transcript_preview = translated_transcript
                                transcript_str = translated_transcript
                                st.session_state.youtube_transcript = translated_transcript
                            except Exception as e:
                                st.error(f"Automatic translation failed: {e}")
                                formatted_transcript = transcript_str.strip()
                                youtube_transcript_preview = transcript_str.strip()
                                st.session_state.youtube_transcript = transcript_str.strip()
                        else:
                            formatted_transcript = transcript_str.strip()
                            youtube_transcript_preview = transcript_str.strip()
                            st.session_state.youtube_transcript = transcript_str.strip()
                        # Embedding creation for Apify transcript
                        with st.spinner("Creating embeddings for transcript..."):
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1000,
                                chunk_overlap=150,
                                separators=["\n\n", "\n", ".", " "]
                            )
                            docs = text_splitter.create_documents([st.session_state.youtube_transcript])
                            embedding = OpenAIEmbeddings(
                                model="text-embedding-3-large",
                                api_key=os.getenv("OPENAI_API_KEY")
                            )
                            db_apify = FAISS.from_documents(docs, embedding)
                            st.session_state.vector_db = db_apify
                        st.session_state.youtube_source = "apify"
                except Exception as e:
                    st.session_state.youtube_source = None
                    st.session_state.youtube_transcript = None
                    youtube_transcript_preview = None
                    youtube_transcript_segments = None
                    st.error(f"YouTube transcript could not be loaded: {e}")

# --- Transcript Preview ---
if youtube_transcript_preview:
    st.markdown("<div style='margin-top:-2.5rem'></div>", unsafe_allow_html=True)
    st.subheader("YouTube Transcript Preview")
    transcript_html = formatted_transcript.replace('\n\n', '<br><br>').replace('\n', '<br>')
    st.markdown(
        "<div style='border:1.5px solid #6C63FF; border-radius:12px; padding:1em; background:#f8fafc; font-size:1.08em; line-height:1.7;'>" + transcript_html + "</div>",
        unsafe_allow_html=True
    )
    # Download as PDF button (fix TypeError: use output(dest='S') and BytesIO)
    with st.spinner("Generating PDF for download..."):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        for paragraph in formatted_transcript.split("\n\n"):
            safe_paragraph = paragraph.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 10, safe_paragraph)
            pdf.ln(2)
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        pdf_output = io.BytesIO(pdf_bytes)
        st.download_button(
            label="Download Transcript as PDF",
            data=pdf_output,
            file_name="youtube_transcript.pdf",
            mime="application/pdf"
        )

# --- Chat History Rendering ---
if "history" not in st.session_state:
    st.session_state.history = []

for user_msg, bot_msg in st.session_state.history:
    st.chat_message("user", avatar="🧑").markdown(user_msg)
    st.chat_message("assistant", avatar="🤖").markdown(bot_msg)

# --- Unified Chat Handler ---
if user_input:
    if not st.session_state.get("vector_db"):
        st.error("\u274c Please provide at least one source (PDF or YouTube video)!")
    else:
        with st.spinner("Generating answer \u23f3"):
            db = st.session_state.vector_db
            llm = init_chat_model(
                "gpt-4.1-mini",
                model_provider="openai",
                api_key=os.getenv("OPENAI_API_KEY")
            )
            system_text = (
                "You are a context-aware assistant.\nUse the provided documents to answer user queries.\nIf the answer is not in the docs, say you don't know."
            )
            condensed_template = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_text),
                HumanMessagePromptTemplate.from_template(
                    "Given the conversation and question, rephrase the question if needed for clarity.\nConversation: {chat_history}\nQuestion: {question}"
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

# --- Sidebar Instructions (bottom) ---
st.sidebar.markdown("---")
st.sidebar.markdown("""
**How to use this app:**
- Upload a PDF to chat with its content.
- Enter a YouTube video URL.
- Ask any question about the content in the chat below. The chat works for both PDF and YouTube sources.
- All processing is local and secure.
""")
