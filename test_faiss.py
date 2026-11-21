import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample text data
texts = [
    "Python is a programming language.",
    "Machine learning is a branch of artificial intelligence.",
    "The Earth revolves around the Sun.",
    "Deep learning uses neural networks."
]

# Convert text into numerical vectors
embeddings = model.encode(texts)
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
index = faiss.IndexFlatL2(384)
index.add(embeddings)

# Search for relevant text
question = "What is machine learning?"
query_vector = model.encode([question])
query_vector = np.array(query_vector).astype("float32")

distances, indices = index.search(query_vector, 2)

print("Question:", question)
print("\nMost relevant results:")

for i in indices[0]:
    print("-", texts[i])
