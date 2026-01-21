# 📄 CV RAG Assistant

> A full-stack Retrieval-Augmented Generation (RAG) application for querying PDF CVs/resumes using natural language.

▶️ **Watch Demo Video**

[![Watch Demo Video](https://cdn.loom.com/sessions/thumbnails/899831f1cb034ef9aac66aff268e74fa-9b59b099789dce76-full-play.gif#t=0.1)](https://www.loom.com/share/899831f1cb034ef9aac66aff268e74fa)


---

### Backend details & setup

See the [Backend README](./ai-rag-backend/README.md) for detailed setup instructions.

### Frontend details & setup

See the [Frontend README](./ai-rag-frontend/README.md) for detailed setup instructions.

---


## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Frontend                                   │
│              (ReactJS + NodeJS + Next.js + AI SDK)                  │
│                                                                     │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐     │
│   │  Chat UI    │───▶│/api/chat-rag│───▶│  Streaming Response │     │
│   │  Component  │    │   Route     │    │ (NDJSON over HTTP)  │     │
│   └─────────────┘    └─────────────┘    └─────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Backend                                       │
│           (Python + FastAPI + LangChain + SQLAlchemy)               │
│                                                                     │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐     │
│   │ RAG Service │───▶│Vector Store │───▶│    LLM (OpenAI)     │     │
│   │             │    │  (pgvector) │    │                     │     │
│   └─────────────┘    └─────────────┘    └─────────────────────┘     │
│                                                                     │
│  RAG Workflow:                                                      │
│   1) Embed query → vector search in pgvector                        │
│   2) Retrieve top-k relevant chunks                                 │
│   3) Augment user prompt with retrieved context                     │
│   4) Generate a grounded answer with the LLM                        │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PostgreSQL + pgvector                            │
│         (Document storage + Vector embeddings for retrieval)        │
└─────────────────────────────────────────────────────────────────────┘
```

---



### 📄 Generate PDFs (CVs) Pipeline

**Script:** `scripts/generate_pdfs.py`

**Steps:**
1. **Generate Structured CV Data** using LangChain with GPT-4o-mini  
2. **Validate Data** with Pydantic models (e.g., name, email, experience, education, etc.)  
3. **Generate Profile Images** using the DALL·E API  
4. **Create PDF Documents** and save them to the `pdf_files/` directory  


### 📥 Ingest PDFs (CVs) Pipeline

**Script:** `scripts/ingest_pdfs.py`

**Steps:**
1. **Load CV PDFs** from the `pdf_files/` directory  
2. **Parse PDF Content** using PyPDF2  
3. **Split Text into Chunks** with LangChain’s RecursiveCharacterTextSplitter (with overlap for context)  
4. **Generate Embeddings** using OpenAI’s `text-embedding-3` model  
5. **Store Vectors and Metadata** in PostgreSQL with the `pgvector` extension  
