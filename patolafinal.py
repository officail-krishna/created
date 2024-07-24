import os
from transformers import AutoTokenizer, AutoModel
import streamlit as st
from streamlit_option_menu import option_menu
from PyPDF2 import PdfReader
from io import BytesIO
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import uuid
from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embeddings(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling to get the sentence embedding
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Initialize session state for context embeddings
if "context_embeddings" not in st.session_state:
    st.session_state.context_embeddings = []

def store_context_embeddings(text):
    embeddings = get_embeddings(text)
    st.session_state.context_embeddings.append(embeddings)

def retrieve_context_embeddings():
    if st.session_state.context_embeddings:
        return torch.cat(st.session_state.context_embeddings, dim=0)
    return None

# Initialize Groq client
client = Groq(
    api_key="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",
)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(BytesIO(pdf_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into smaller chunks
def split_text_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Function to retrieve relevant text from the PDF content using TF-IDF
def retrieve_relevant_text(pdf_text, query):
    # Split the PDF text into chunks
    chunks = split_text_into_chunks(pdf_text)
    
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer().fit_transform(chunks + [query])
    vectors = vectorizer.toarray()
    
    # Compute cosine similarity between the query and the chunks
    cosine_similarities = cosine_similarity(vectors[-1:], vectors[:-1]).flatten()
    
    # Get the most relevant chunk
    most_relevant_index = np.argmax(cosine_similarities)
    relevant_text = chunks[most_relevant_index]
    
    return relevant_text

# Function to transcribe audio using Groq
def transcribe_audio(file, language):
    transcription = client.audio.transcriptions.create(
        file=(file.name, file.read()),
        model="whisper-large-v3",
        prompt="Specify context or spelling",  # Optional
        response_format="json",  # Optional
        language=language,  # Optional
        temperature=0.2  # Optional
    )
    return transcription.text

# Get the working directory every time we run the file
working_dir = os.path.dirname(os.path.abspath(__file__))

# Setting page configuration
st.set_page_config(
    page_icon="✨",
    page_title="PDF Patola",
    layout="centered"
)

# Initialize session state for selected model
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama-3.1-70b-versatile"

with st.sidebar:
    selected = option_menu(
        menu_title="Choose Functionality",
        options=["PDF Reader", "Speech Recognition"],
        icons=["file-earmark-pdf", "mic"],
        default_index=0
    )

# Model Selection Section (Always Visible)
st.sidebar.subheader("Select Model for PDF Reader")
model_options = ["llama-3.1-70b-versatile", "llama-3.1-405b-reasoning", "gemma-7b-it"]
selected_model = st.sidebar.selectbox("Choose a model", model_options, index=model_options.index(st.session_state.selected_model))

if selected_model != st.session_state.selected_model:
    st.session_state.selected_model = selected_model
    st.experimental_rerun()

# Custom CSS and JavaScript
st.markdown("""
    <style>
        body {
            font-family: 'Helvetica', sans-serif;
            background: url('https://www.transparenttextures.com/patterns/stardust.png'), url('https://www.transparenttextures.com/patterns/flowers.png');
            background-size: cover;
        }
        .chat-message {
            position: relative;
            padding: 10px;
            margin: 10px 0;
            border-radius: 10px;
            animation: fadeIn 0.5s ease-in-out;
        }
        .chat-message.user {
            background-color: #DCF8C6;
            text-align: right;
        }
        .chat-message.assistant {
            background-color: #F1F0F0;
            color: #000000;
            text-align: left;
        }
        .chat-message .arrow {
            width: 0;
            height: 0;
            border-style: solid;
            position: absolute;
        }
        .chat-message.user .arrow {
            border-width: 10px 0 10px 10px;
            border-color: transparent transparent transparent #DCF8C6;
            right: -10px;
            top: 10px;
        }
        .chat-message.assistant .arrow {
            border-width: 10px 10px 10px 0;
            border-color: transparent #F1F0F0 transparent transparent;
            left: -10px;
            top: 10px;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        .copy-button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 5px 10px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 12px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
        }
        .copy-button:hover {
            background-color: #45a049;
        }
        .sidebar .option-menu {
            animation: slideIn 0.5s ease-in-out;
        }
        @keyframes slideIn {
            from { transform: translateX(-100%); }
            to { transform: translateX(0); }
        }
        .chat-input-container {
            position: fixed;
            bottom: 0;
            width: 100%;
            background: white;
            padding: 10px;
            box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
        }
    </style>
    <script>
        function copyToClipboard(text) {
            navigator.clipboard.writeText(text).then(function() {
                alert('Copied to clipboard');
            }, function(err) {
                console.error('Could not copy text: ', err);
            });
        }
    </script>
""", unsafe_allow_html=True)

# PDF Reader Page
if selected == "PDF Reader":
    st.title("PDF Patola")
    st.write("Chai Piyoge ☕????")
    st.write('Pro tip - always use "clean chat history"')

    # Generate a unique session ID for each user
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    session_id = st.session_state.session_id

    # File uploader for multiple PDFs
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        # Extract text from each PDF
        pdf_texts = [extract_text_from_pdf(file) for file in uploaded_files]
        combined_pdf_text = "\n\n".join(pdf_texts)

        # Initialize session state for chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Placeholder for chat messages
        chat_placeholder = st.empty()

        # Display chat history
        with chat_placeholder.container():
            for role, message in st.session_state.chat_history:
                if role == "user":
                    st.markdown(f'<div class="chat-message user"><div class="message">{message}</div><div class="arrow user"></div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message assistant"><div class="arrow assistant"></div><div class="message">{message}</div></div>', unsafe_allow_html=True)

        # Input box at the bottom with embedded send button
        user_input = st.chat_input("Ask a question about the PDF content:")
        if user_input:
            # Retrieve relevant text from the combined PDF content
            relevant_text = retrieve_relevant_text(combined_pdf_text, user_input)

            # Store context embeddings
            store_context_embeddings(relevant_text)

            # Retrieve context embeddings
            context_embeddings = retrieve_context_embeddings()

            # Generate response using the Groq client
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. The following is the relevant content of the PDF: " + combined_pdf_text + " act like you are a pdf"
                    },
                    {
                        "role": "user",
                        "content": user_input,
                    }
                ],
                model=st.session_state.selected_model,
            )
            response = chat_completion.choices[0].message.content

            # Update chat history
            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("assistant", response))

            # Display the updated chat history
            with chat_placeholder.container():
                for role, message in st.session_state.chat_history:
                    if role == "user":
                        st.markdown(f'<div class="chat-message user"><div class="message">{message}</div><div class="arrow user"></div></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-message assistant"><div class="arrow assistant"></div><div class="message">{message}</div></div>', unsafe_allow_html=True)

            # Add copy button
            # st.markdown(f'<button class="copy-button" onclick="copyToClipboard(`{response}`)">Copy</button>', unsafe_allow_html=True)
    # Add the "Clear Chat" button below the chatbox
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.context_embeddings = []
        st.session_state.uploaded_files = []
        st.experimental_rerun()
# Speech Recognition Page
if selected == "Speech Recognition":
    st.title("Speech to Text 🎤")

    uploaded_audio = st.file_uploader("Upload an audio file..", type=["m4a", "mp3", "wav"])

    # Add a dropdown menu for language selection
    language = st.selectbox(
        "Select language for transcription:",
        ("en", "hi", "es", "fr", "de", "ja", "ru")
    )
    if st.button("Transcribe Audio"):
        if uploaded_audio is not None:
            transcription_text = transcribe_audio(uploaded_audio, language)
            st.write("Transcription:")
            st.write(transcription_text)
            # Add copy button for transcription text
            st.markdown(f'<button class="copy-button" onclick="copyToClipboard(`{transcription_text}`)">Copy</button>', unsafe_allow_html=True)
        else:
            st.write("Please upload an audio file to transcribe.")
