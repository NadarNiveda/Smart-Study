import pickle
import faiss
import subprocess
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Academic RAG Backend")

# Load vector store + chunks
index = faiss.read_index("vector_store.faiss")
chunks = pickle.load(open("chunks.pkl", "rb"))

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Distance threshold (lower = stricter, 1.0–1.5 is good)
THRESHOLD = 1.3
TOP_K = 5


class Query(BaseModel):
    question: str


def query_ollama(prompt):
    """Send prompt to local Ollama LLM and return result (UTF-8 Safe)"""

    process = subprocess.Popen(
        ["ollama", "run", "orca-mini:3b"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    out, err = process.communicate(prompt.encode("utf-8"))
    return out.decode("utf-8").strip()



@app.post("/ask")
def ask(query: Query):

    # 1. Embed the user question
    q_embed = embed_model.encode([query.question]).astype("float32")

    # 2. Search in FAISS
    distances, ids = index.search(q_embed, TOP_K)

    print("\nQUESTION:", query.question)
    print("DISTANCES:", distances)
    print("IDS:", ids)

    # 3. Filter good chunks only
    good_chunks = []

    for dist, idx in zip(distances[0], ids[0]):
        if idx == -1:
            continue

        if dist <= THRESHOLD:
            good_chunks.append(chunks[idx])

    # If nothing relevant → return NOT FOUND
    if len(good_chunks) == 0:
        return {"answer": "Not found in book."}

    # 4. Join context
    context = "\n\n".join(good_chunks[:3])

    # 5. Stronger, forced prompt
    prompt = f"""
You are a strict academic assistant.

You MUST answer only from the given CONTEXT.
If the answer is missing, respond exactly with: Not found in book.
Do NOT use outside knowledge. Do NOT guess.

CONTEXT:
{context}

QUESTION:
{query.question}

ANSWER (only from context):
"""

    # 6. Run Ollama
    try:
        result = query_ollama(prompt)

        if result.strip() == "":
            result = "Not found in book."

    except subprocess.TimeoutExpired:
        result = "LLM timeout. Try a shorter question."

    return {"answer": result}
