# RAG Document Chatbot

A beginner-friendly **Retrieval-Augmented Generation (RAG)** application built with Python.

Upload a PDF document, ask questions about its content, and get answers grounded in the uploaded document.

This project is part of the **AI Projects Series** from **Karthik's Show**.

## Features

- Upload a PDF document
- Extract text using `pypdf`
- Split document text into overlapping chunks
- Generate embeddings using Sentence Transformers
- Store and search embeddings with FAISS
- Retrieve the most relevant document sections for each question
- Generate grounded answers using Groq
- Chat-style interface using Streamlit
- Maintain conversation history with Streamlit session state
- Clear chat history when a different PDF is uploaded
- Handle PDFs with no readable text
- Handle missing API keys and model/API errors
- Avoid answering from outside the uploaded document

## Tech Stack

- Python
- Streamlit
- pypdf
- Sentence Transformers
- FAISS
- Groq
- python-dotenv

## How RAG Works in This Project

```text
PDF Upload
    ↓
Text Extraction
    ↓
Text Chunking
    ↓
Document Embeddings
    ↓
FAISS Vector Index
    ↓
User Question
    ↓
Question Embedding
    ↓
Similarity Search
    ↓
Relevant Document Chunks
    ↓
Groq LLM
    ↓
Grounded Answer
```

The language model does not receive the entire PDF.

Instead, FAISS retrieves the most relevant document chunks for the user's question, and only those chunks are passed to the Groq model as context.

## Project Structure

```text
rag-document-chatbot/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── rag_demo_company_handbook.pdf
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/KarthiksShow/rag-document-chatbot.git
cd rag-document-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_actual_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

You can use `.env.example` as a reference.

> Do not commit your `.env` file or Groq API key to GitHub.

## Run the Application

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## How to Use

1. Upload a text-based PDF.
2. Wait for the document to be processed.
3. Enter a question in the chat box.
4. The application retrieves the most relevant document sections using FAISS.
5. Groq generates an answer using only the retrieved document context.
6. Continue asking questions from the same document.

If a different PDF is uploaded, the previous chat history is cleared.

## Sample PDF

A sample PDF is included:

```text
rag_demo_company_handbook.pdf
```

You can use it to test the complete RAG workflow.

Example questions:

```text
How many days per week can eligible employees work remotely?
```

```text
What is the annual learning budget?
```

```text
When must a security incident be reported?
```

```text
Can confidential customer data be entered into public AI tools?
```

You can also test grounding with a question whose answer is not present in the document:

```text
Who is the CEO of the company?
```

The chatbot should respond that it could not find the information in the document instead of inventing an answer.

## RAG Components Used

### 1. Text Extraction

`pypdf` reads the uploaded PDF and extracts readable text from each page.

### 2. Chunking

The extracted document text is split into overlapping chunks.

Current settings:

```text
Chunk size: 180 words
Chunk overlap: 40 words
```

Overlap helps preserve context when information falls near a chunk boundary.

### 3. Embeddings

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Document chunks are converted into normalized numerical embeddings.

### 4. Vector Search

FAISS stores the document embeddings and performs similarity search.

The application retrieves the top 3 most relevant chunks for each user question.

### 5. Answer Generation

The retrieved chunks are combined into context and sent to the Groq language model.

The system prompt instructs the model to:

- Answer only using the supplied document context
- Avoid outside knowledge
- Clearly say when the requested information is not available
- Keep answers clear and concise

## Important Notes

- This project works best with PDFs that contain selectable text.
- Scanned or image-only PDFs require OCR, which is not included in this version.
- The embedding model is downloaded the first time the application runs.
- The current beginner-friendly implementation rebuilds document embeddings and the FAISS index during Streamlit reruns.
- For production use, document processing and vector indexes should be cached or persisted more efficiently.
- This demo supports one uploaded PDF at a time.

## Learning Outcomes

By building this project, you can understand:

- What Retrieval-Augmented Generation is
- Why documents are split into chunks
- What embeddings represent
- How vector similarity search works
- How FAISS retrieves relevant context
- How retrieved context is passed to an LLM
- How grounding can reduce unsupported answers
- How to build a simple AI chat interface with Streamlit

## YouTube Tutorial

This project is explained step by step in Tamil on **Karthik's Show**.

The tutorial covers:

- Project setup
- PDF text extraction
- Document chunking
- Embedding generation
- FAISS similarity search
- Groq answer generation
- Streamlit chat interface
- Final RAG testing

## Channel

**Karthik's Show**

Learn. Build. Grow.

## License

This project is intended for learning and demonstration purposes.
