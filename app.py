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

st.title("📄 RAG Document Chatbot")

st.write(
    "Upload a PDF document and ask questions "
    "based on its content."
)

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
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