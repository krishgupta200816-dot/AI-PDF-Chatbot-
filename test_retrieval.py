
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# 1. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Extract PDF text
def read_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


# 3. Divide text into chunks
def make_chunks(pages, chunk_size=800, overlap=120):
    chunks = []

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            chunks.append({
                "page": page["page"],
                "text": text[start:end]
            })

            if end == len(text):
                break

            start += chunk_size - overlap

    return chunks


# 4. Load your PDF
pdf_path = input("Enter PDF path: ")
pages = read_pdf(pdf_path)
chunks = make_chunks(pages)

# 5. Convert chunks into embeddings
texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(texts)
embeddings = np.array(embeddings).astype("float32")

# 6. Create FAISS index
index = faiss.IndexFlatL2(384)
index.add(embeddings)

print("Total chunks indexed:", index.ntotal)

# 7. Ask a question
question = input("\nAsk something about your PDF: ")

query_vector = model.encode([question])
query_vector = np.array(query_vector).astype("float32")

distances, indices = index.search(query_vector, 3)

# 8. Display relevant chunks
print("\nRelevant PDF information:")

for i in indices[0]:
    print("\nPage:", chunks[i]["page"])
    print(chunks[i]["text"])
    print("-" * 50)
