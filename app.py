import os
import re
import streamlit as st
import faiss
from dotenv import load_dotenv
from pypdf import PdfReader, errors
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import errors

# Read private settings from the env file
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

st.set_page_config(page_title="AI PDF Chatbot", page_icon="◈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

.stApp {
    background-color: #F7F7F4;
    color: #172B4D;
    font-family: 'Manrope', sans-serif;
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}

.block-container {
    max-width: 1150px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #172B4D;
    font-weight: 700;
    letter-spacing: -0.5px;
}

p, label, span {
    color: #44546A;
}

.stButton > button {
    background-color: #3B4B64 !important;
    color: #F7F7F4 !important;
    border: 1px solid #3B4B64 !important;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button p,
.stButton > button span {
    color: #F7F7F4 !important;
}

.stButton > button:hover {
    background-color: #2A374B !important;
    border-color: #2A374B !important;
    color: #F7F7F4 !important;
}
}

.stTextInput input {
    background-color: #FFFFFF;
    border: 1px solid #D9DFE7;
    border-radius: 9px;
    color: #172B4D;
}

.stFileUploader {
    background-color: #FFFFFF;
    border: 1px dashed #AAB8CA;
    border-radius: 12px;
    padding: 12px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border: 1px solid #E4E7EC;
    border-radius: 12px;
}

hr {
    border-color: #E4E7EC;
}

</style>
""", unsafe_allow_html=True)

st.title("AI-Powered PDF Chatbot")

if not API_KEY:
    st.error("Gemini API key is missing. Add GEMINI_API_KEY to your .env file.")
    st.stop()

client = genai.Client(api_key=API_KEY)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedding_model()

def read_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r'\s+', ' ', text).strip()
        if text:
            pages.append({"page": number, "text": text})
    return pages

def make_chunks(pages, chunk_size=800, overlap=120):
    chunks = []
    for item in pages:
        text = item["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            piece = text[start:end].strip()
            if piece:
                chunks.append({"page": item["page"], "text": piece})
            start += chunk_size - overlap
    return chunks

def build_search_index(chunks):
    texts = [item["text"] for item in chunks]
    vectors = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index

def retrieve(question, chunks, index, top_k=4):
    query_vector = embedder.encode([question], convert_to_numpy=True, show_progress_bar=False).astype("float32")
    count = min(top_k, len(chunks))
    scores, positions = index.search(query_vector, count)
    results = []
    for position in positions[0]:
        if position != -1:
            results.append(chunks[int(position)])
    return results


def ask_gemini(prompt):
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        return response.text

    except errors.APIError as error:
        if error.code in [500, 502, 503, 504, 429]:
            return (
                "Gemini is temporarily unavailable or busy. "
                "Please wait a little and try again."
            )

        return f"Gemini API error: {error}"


def answer_question(question, matches):
    context = "\n\n".join([f"Page {m['page']}: {m['text']}" for m in matches])
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not present, say: "The information was not found in the uploaded document."
Do not guess or invent facts.

Context:
{context}

Question: {question}"""
    return ask_gemini(prompt)

def create_quiz(all_text, number_of_questions):
  prompt = (
      f"Create exactly {number_of_questions} multiple-choice questions "
      "from the PDF text below. Each question must have four options"
      " (A-D). Include a separate answer key at the end.\n\n"
      + all_text[:12000]
  )
  return ask_gemini(prompt)

def create_summary(all_text):
    prompt = f"""
    You are an expert document summarisation assistant.

    Summarise the following PDF content in a clear and organised way.

    Follow this structure:

    1. Overview
    2. Main Topics
    3. Important Concepts
    4. Key Points
    5. Conclusion

    Use simple language.
    Include important details.
    Do not invent information that is not in the document.

    PDF Content:
    {all_text[:12000]}
    """

    return ask_gemini(prompt)


# --- Streamlit UI ---
uploaded_files = st.file_uploader("Choose PDF file(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_pages = []
    for uploaded in uploaded_files:
        try:
            pages = read_pdf(uploaded)
            all_pages.extend(pages)
        except Exception as error:
            st.error(f"Could not read {uploaded.name}: {error}")
            
    if not all_pages:
        st.warning("No selectable text was found. This may be a scanned PDF.")
    else:
        chunks = make_chunks(all_pages)
        if not chunks:
            st.warning("No readable text was found in the uploaded PDF.")
        else:
            full_text = "\n\n".join(item["text"] for item in all_pages)
            with st.spinner("Preparing document search..."):
                index = build_search_index(chunks)
            st.success(f"Ready: {len(all_pages)} text page(s), {len(chunks)} chunk(s).")
            st.divider()

        st.subheader("📄 PDF Summary")

        if st.button("Generate Summary"):
            with st.spinner("Generating your PDF summary..."):
                summary = create_summary(full_text)

            st.markdown("### Document Summary")
            st.markdown(summary)
            
            st.subheader("Ask a question")
            question = st.text_input("Type your question about the PDF")
            if st.button("Get answer"):
                if not question.strip():
                    st.warning("Please type a question first.")
                else:
                    with st.spinner("Finding the answer..."):
                        matches = retrieve(question, chunks, index)
                        answer = answer_question(question, matches)
                    st.markdown("### Answer")
                    st.write(answer)
                    st.markdown("### Sources")
                    for m in matches:
                        st.divider()
                        st.write(f"Page {m['page']}")
                        st.write(m['text'])

            st.divider()

            st.subheader("Generate MCQ Quiz")

            full_text = "\n\n".join(
                item["text"] for item in all_pages
            )

            question_count = st.selectbox(
                "Number of questions",
                [5, 10, 15]
            )

            if st.button("Generate Quiz"):
                with st.spinner("Generating your quiz..."):
                    quiz = create_quiz(
                        full_text,
                        question_count
                    )

                st.markdown("### Your MCQ Quiz")
                st.write(quiz)
