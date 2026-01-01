import os
import pickle
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

PDF_FOLDER = "data/"
CHUNK_SIZE = 500  # words per chunk

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text, size=CHUNK_SIZE):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size):
        chunk = " ".join(words[i:i+size])
        chunks.append(chunk)
    return chunks

def build_vector_store(pdf_folder=PDF_FOLDER):
    all_chunks = []
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    print(f"Found PDFs: {pdf_files}")

    for pdf in pdf_files:
        print(f"Processing {pdf}...")
        text = load_pdf(os.path.join(pdf_folder, pdf))
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    print("Embedding chunks with MiniLM...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(all_chunks)

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save FAISS index and chunks
    faiss.write_index(index, "vector_store.faiss")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print("Vector store created successfully!")

if __name__ == "__main__":
    build_vector_store()
