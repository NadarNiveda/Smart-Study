# 📚 Academic Book-Based Q&A (RAG Backend)

This project is a **book-based Question Answering system** built using **Retrieval-Augmented Generation (RAG)**.  
The system answers questions **strictly from the provided textbooks (PDFs)** and responds with **detailed, easy-to-understand explanations** without using outside knowledge.

---

## 🚀 Features
- ✅ Answers **only from book content**
- ✅ Rejects questions not found in the book
- ✅ Detailed explanations while preserving original meaning
- ✅ Fast semantic search using **FAISS**
- ✅ Local LLM inference using **Ollama**
- ✅ REST API using **FastAPI**
- ✅ Ready to integrate with **React / Android frontend**

---

## 🧠 Tech Stack
**Backend**
- Python 3.10+
- FastAPI
- FAISS
- Sentence Transformers (MiniLM)
- Ollama (orca-mini:3b)

**Frontend (separate)**
- React.js / Android Studio (API-based integration)

---

## 📁 Project Structure
rag_model/
│
├── app.py # FastAPI backend
├── process_data.py # PDF → chunks → embeddings → FAISS
├── requirements.txt # Python dependencies
├── vector_store.faiss # FAISS vector index
├── chunks.pkl # Book text chunks
├── data/ # PDF textbooks
│ ├── CN.pdf
│ ├── dbms.pdf
│ ├── ds.pdf
│
├── .gitignore
└── README.md
