# AI-PDF-Chatbot

An AI-powered PDF document assistant built using Python, Streamlit, Google Gemini, FAISS, and Sentence Transformers.

**Live Application:** [Open AI-PDF-Chatbot](https://imj5yfijxhm6ismkkmhkkx.streamlit.app/)

## Features

- Upload one or multiple PDF documents.
- Extract and process text page by page.
- Split documents into overlapping text chunks.
- Generate semantic embeddings using Sentence Transformers.
- Retrieve relevant document information using FAISS.
- Ask questions about uploaded PDFs using Google Gemini.
- Display source page references for answers.
- Generate structured PDF summaries.
- Generate multiple-choice quizzes with answer keys.
- Handle empty, unreadable, and invalid PDF inputs.

## Technologies Used

- Python
- Streamlit
- Google Gemini API
- Sentence Transformers
- FAISS
- PyPDF
- python-dotenv

## How It Works

1. User uploads one or more PDF documents.
2. Text is extracted page by page.
3. Extracted text is cleaned and divided into overlapping chunks.
4. Sentence Transformers converts chunks into numerical embeddings.
5. FAISS stores and searches the embeddings.
6. Relevant chunks are retrieved for a user's question.
7. Gemini generates an answer using the retrieved document context.
8. Users can also generate summaries and MCQ quizzes.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/krishgupta200816-dot/AI-PDF-Chatbot-.git
cd AI-PDF-Chatbot-
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Linux/WSL:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project directory:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
```

Never upload your actual API key to GitHub.

### 5. Run the application

```bash
streamlit run app.py
```

## Testing

The application was tested against the following cases:

- T01: Valid PDF upload.
- T02: Rejection of unsupported file types.
- T03: Question answering with source references.
- T04: Handling questions whose answers are absent from the document.
- T05: PDF summary generation.
- T06: MCQ quiz generation with answer key.
- T07: Handling PDFs without readable text.
- T08: Missing API key error handling.

All eight test cases have been reported as passing.

## Project Status

The core application features and functional testing are complete.

## Security

- API keys are stored in environment variables.
- `.env` is excluded from Git tracking.
- Never share private API credentials.

## Author

Developed as a Python and AI-based document assistant project.
