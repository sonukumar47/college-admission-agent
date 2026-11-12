# 🎓 College Admission Agent (RAG-Based)
**IBM Granite × watsonx.ai × ChromaDB**

A Retrieval-Augmented Generation (RAG) powered College Admission Agent built on **IBM Cloud Lite (free tier)**. Prospective students can ask natural language questions about admission policies, courses, fees, deadlines, and FAQs — and receive accurate, grounded answers instantly.

---

## Architecture

```
Student Question
      │
      ▼
[ChromaDB Vector Store]  ←── [Knowledge Base JSON files]
      │  (semantic retrieval)
      ▼
[IBM Granite LLM]  ←── IBM watsonx.ai (FREE Lite: 300K tokens/month)
      │  (grounded answer generation)
      ▼
  Answer + Sources
```

**Stack:**
| Layer | Technology | Cost |
|---|---|---|
| LLM | IBM Granite (`ibm/granite-13b-instruct-v2`) | FREE (Lite plan) |
| Vector Store | ChromaDB (local persistent) | FREE |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | FREE |
| Web UI | Flask | FREE |
| IBM Cloud | watsonx.ai Runtime Lite | FREE |

---

## Prerequisites

- Python 3.9+
- IBM Cloud account (free — no credit card required)
- IBM watsonx.ai Runtime **Lite** plan instance

---

## Quick Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure IBM Cloud credentials

Edit `.env` and fill in your credentials:

```env
WATSONX_API_KEY=your_ibm_cloud_api_key_here
WATSONX_PROJECT_ID=your_watsonx_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

**How to get credentials:**
1. Go to [IBM Cloud](https://cloud.ibm.com) → Sign up for a free Lite account
2. Create a **watsonx.ai Runtime** service (Lite plan, free)
3. Get your **API key**: IBM Cloud → Manage → IAM → API Keys → Create
4. Get your **Project ID**: Open your watsonx.ai project → Manage tab → copy Project ID

### 3. Run the Web App
```bash
python app.py
```
Open http://localhost:5000 in your browser.

### 4. Run the CLI
```bash
python cli.py
```

---

## Project Structure

```
college-admission-agent/
├── .env                          # IBM Cloud credentials (edit this)
├── requirements.txt              # Python dependencies
├── app.py                        # Flask web application
├── cli.py                        # Command-line interface
├── src/
│   ├── agent.py                  # Agent initializer
│   ├── granite_client.py         # IBM Granite LLM client
│   ├── knowledge_base.py         # ChromaDB vector store loader
│   └── rag_engine.py             # RAG retrieval + generation engine
├── data/
│   ├── knowledge_base/           # Source JSON documents
│   │   ├── admission_policies.json
│   │   ├── courses.json
│   │   ├── fee_structure.json
│   │   ├── deadlines.json
│   │   └── faqs.json
│   └── vectorstore/              # ChromaDB persistent store (auto-created)
└── templates/
    └── index.html                # Chat web UI
```

---

## Sample Questions

- *"What are the undergraduate admission requirements?"*
- *"What is the tuition fee for international students?"*
- *"When is the application deadline for fall 2025?"*
- *"What scholarships are available for freshmen?"*
- *"Tell me about the Computer Science program"*
- *"Is housing guaranteed for freshmen?"*
- *"What documents do international students need?"*
- *"What is the MBA program fee?"*

---

## IBM Cloud Free Tier Details

| Service | Free Limit | Notes |
|---|---|---|
| watsonx.ai Runtime Lite | **300,000 tokens/month** | Resets monthly |
| watsonx.ai Studio Lite | **10 CUH/month** | For prompt testing |
| IBM Cloud Account | Always free | No credit card needed |

At ~600 tokens per Q&A, the free tier supports **~500 student interactions/month**.

---

## Extending the Knowledge Base

Add new JSON files to `data/knowledge_base/` following this format:
```json
{
  "college": "Greenfield University",
  "documents": [
    {
      "id": "unique_id",
      "category": "your_category",
      "title": "Document Title",
      "content": "Full text content..."
    }
  ]
}
```

Delete `data/vectorstore/` and restart the app to re-index.
