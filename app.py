import os

import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq
import faiss

st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="📄",
    layout="wide"
)

st.markdown(
    """
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #ecf7ff 0%, #f5ebff 35%, #fff6d9 100%);
            color: var(--text-color, #1d2340);
        }

        .stApp {
            background: linear-gradient(135deg, #ecf7ff 0%, #f5ebff 35%, #fff6d9 100%);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }

        .hero-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(228,242,255,0.9), rgba(242,231,255,0.9));
            border: 1px solid rgba(90, 112, 255, 0.25);
            border-radius: 24px;
            padding: 1.6rem 1.7rem;
            box-shadow: 0 18px 40px rgba(120, 132, 255, 0.18);
            margin-bottom: 1.4rem;
        }

        .neon-title {
            font-size: 2.5rem;
            line-height: 1.15;
            color: var(--text-color, #2a2fd8);
            text-shadow: 0 0 10px rgba(89, 138, 255, 0.28), 0 0 18px rgba(255, 118, 196, 0.18);
            margin-bottom: 0.35rem;
            letter-spacing: 0.04em;
        }

        .subtext {
            color: var(--text-color, #334164);
            font-size: 1.04rem;
            margin-top: 0.25rem;
        }

        .glow-badge {
            display: inline-block;
            padding: 0.42rem 0.8rem;
            border-radius: 999px;
            background: linear-gradient(135deg, #5d8bff, #8f6cff, #ff7edb);
            border: 1px solid rgba(255,255,255,0.4);
            color: #ffffff;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            font-size: 0.72rem;
            box-shadow: 0 8px 20px rgba(139, 110, 255, 0.25);
        }

        .section-box {
            background: rgba(255,255,255,0.7);
            border: 1px solid rgba(96, 116, 177, 0.22);
            border-radius: 18px;
            padding: 1rem 1.2rem;
            box-shadow: 0 10px 24px rgba(129, 149, 255, 0.10);
        }

        [data-testid="stFileUploader"] > div {
            background: rgba(255,255,255,0.75);
            border: 1px solid rgba(100, 120, 255, 0.28);
            border-radius: 16px;
            box-shadow: 0 10px 24px rgba(137, 145, 255, 0.08);
        }

        [data-testid="stChatMessage"] {
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(105, 127, 255, 0.18);
            border-radius: 18px;
        }

        .stChatInput {
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(103, 113, 255, 0.25);
            border-radius: 16px;
        }

        .stMarkdown p, .stMarkdown li {
            color: #202b46;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY is missing. "
        "Please add it to the .env file."
    )
    st.stop()

groq_client = Groq(
    api_key=GROQ_API_KEY
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

embedding_model = load_embedding_model()

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    extracted_pages = []

    for page in pdf_reader.pages:
        page_text = page.extract_text()

        if page_text:
            extracted_pages.append(page_text)

    document_text = "\n".join(extracted_pages)

    return document_text, len(pdf_reader.pages)

def split_text_into_chunks(
    text,
    chunk_size=180,
    chunk_overlap=40
):
    words = text.split()
    chunks = []

    step_size = chunk_size - chunk_overlap

    for start_index in range(
        0,
        len(words),
        step_size
    ):
        end_index = start_index + chunk_size

        chunk_words = words[
            start_index:end_index
        ]

        chunk_text = " ".join(chunk_words)

        if chunk_text.strip():
            chunks.append(chunk_text)

    return chunks

def create_chunk_embeddings(
    chunks,
    model
):
    embeddings = model.encode_document(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings

def create_faiss_index(embeddings):
    embedding_dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        embedding_dimension
    )

    index.add(
        embeddings.astype("float32")
    )

    return index

def retrieve_relevant_chunks(
    question,
    model,
    index,
    chunks,
    top_k=3
):
    question_embedding = model.encode_query(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = (
        question_embedding
        .astype("float32")
        .reshape(1, -1)
    )

    scores, indices = index.search(
        question_embedding,
        top_k
    )

    relevant_chunks = []

    for chunk_index in indices[0]:
        if chunk_index != -1:
            relevant_chunks.append(
                chunks[chunk_index]
            )

    return relevant_chunks

def generate_answer(
    question,
    relevant_chunks,
    client,
    model_name
):
    context = "\n\n".join(
        relevant_chunks
    )

    system_prompt = """
You are a document question-answering assistant.

Answer the user's question using only the context
provided from the uploaded document.

If the answer is not available in the context, say:
"I could not find that information in the document."

Do not use outside knowledge.
Keep the answer clear and concise.
"""

    user_prompt = f"""
Document context:

{context}

Question:

{question}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0
    )

    answer = (
        response.choices[0]
        .message.content
    )

    return answer

st.markdown(
    """
    <div class="hero-card">
        <div class="glow-badge">AI Document Assistant</div>
        <h1 class="neon-title">Hey Sangeeth, I am your document helper.</h1>
        <p class="subtext">
            Drop in a PDF and I’ll help you find the answers hidden inside it with a fast, grounded RAG workflow.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-box">
        <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.3rem;">
            <span style="color: var(--primary-color, #3f3df5); font-size: 1.3rem;">📄</span>
            <strong style="color: var(--text-color, #1d2340); font-size: 1.08rem; font-weight: 700;">Upload your document</strong>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    if (
        st.session_state.uploaded_file_name
        != uploaded_file.name
    ):
        st.session_state.messages = []

        st.session_state.uploaded_file_name = (
            uploaded_file.name
        )
    st.success(
        f"Uploaded successfully: {uploaded_file.name}"
    )

    document_text, page_count = (
            extract_text_from_pdf(uploaded_file)
        )

    if not document_text.strip():
        st.error(
            "No readable text was found in this PDF. "
            "It may be scanned or image-based."
        )
        st.stop()

    #word_count = len(document_text.split())

    document_chunks = split_text_into_chunks(document_text)

    #chunk_count = len(document_chunks)

    with st.spinner(
    "Creating document embeddings..."
    ):
        chunk_embeddings = (
            create_chunk_embeddings(
                document_chunks,
                embedding_model
            )
        )

    faiss_index = create_faiss_index(
    chunk_embeddings
    )

    for message in st.session_state.messages:
        with st.chat_message(
            message["role"]
        ):
            st.markdown(
                message["content"]
            )

    user_question = st.chat_input(
    "Ask a question about the document"
    )

    if user_question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        relevant_chunks = (
            retrieve_relevant_chunks(
                user_question,
                embedding_model,
                faiss_index,
                document_chunks,
                top_k=3
            )
        )
        try:
          with st.chat_message("assistant"):
            with st.spinner(
                "Searching the document..."
            ):
                answer = generate_answer(
                    user_question,
                    relevant_chunks,
                    groq_client,
                    GROQ_MODEL
                )

            if not answer or not answer.strip():
                st.error(
                    "The model returned an empty response."
                )
                st.stop()

            st.markdown(answer)

        except Exception as error:
            st.error(
                "Unable to generate an answer. "
                "Please check your API key, model "
                "configuration, and internet connection."
            )

            st.caption(str(error))
            st.stop()

        st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
        )

        #st.subheader("Answer")

        #st.write(answer)

    # Display document statistics in three columns.
    #info_col1, info_col2, info_col3 = st.columns(3)

    #with info_col1:
        #st.metric("Pages", page_count)

    #with info_col2:
        #st.metric("Words", word_count)

    #with info_col3:
        #st.metric("Chunks", chunk_count)

    # Show the full extracted text in a collapsible expander.
    #with st.expander("Preview extracted text"):
        #st.text_area(
            #"Document text",
            #document_text,
            #height=300,
            #disabled=True
        #)

    # Show a preview of the first few document chunks.
    #with st.expander("Preview document chunks"):
        #preview_count = min(
            #3,
            #len(document_chunks)
        #)

        #for index in range(preview_count):
            #st.markdown(
                #f"**Chunk {index + 1}**"
            #)

            #st.write(
                #document_chunks[index]
            #)

            #st.divider()