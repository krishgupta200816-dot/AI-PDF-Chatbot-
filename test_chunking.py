from pypdf import PdfReader


def read_pdf(pdf_path):
  reader = PdfReader(pdf_path)
  pages = []
  for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text()
    if text:
      pages.append({"page": page_number, "text": text})
  return pages


def make_chunks(pages, chunk_size=800, overlap=120):
  chunks = []
  for page in pages:
    text = page["text"]
    start = 0
    while start < len(text):
      # Use min() so we don't go past the actual end of the text string
      end = min(start + chunk_size, len(text))
      chunk_text = text[start:end]
      chunks.append({"page": page["page"], "text": chunk_text})

      # If we've reached the end of the text, stop the loop for this page
      if end == len(text):
        break

      start += chunk_size - overlap
  return chunks


pdf_path = input("Enter PDF path: ")
pages = read_pdf(pdf_path)
chunks = make_chunks(pages)

print("Total pages:", len(pages))
print("Total chunks:", len(chunks))
print("\nFirst chunk:")
print(chunks[0]["text"])
print("\nPage number:", chunks[0]["page"])