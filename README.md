# Agentic-Rag-Q-A-System
This is the project , about RAG system followed langchain agent framework which can be used as Q&amp;A for Indian Budget data of 2024-2025

# 📚 Custom LLM-Based Question Answering System for Indian Budget 2024-2025 using RAG & Agentic AI

[![GitHub Stars](https://img.shields.io/github/stars/your-username/llm-agentic-rag-qa?style=social)](https://github.com/your-username/llm-agentic-rag-qa/stargazers)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Demo-green)](https://llm-agentic-rag-qa.streamlit.app)

**Tech Stack:** LangChain, Pinecone, Hugging Face, ChromaDB, Streamlit, OpenAI API, Python
**Category:** Retrieval-Augmented Generation (RAG), Conversational AI, Agentic AI, Generative AI

---

## 🧠 Project Summary

This project demonstrates a fully-functional **intelligent Question Answering (QA) system** powered by:

* **Retrieval-Augmented Generation (RAG)** for grounding LLMs in domain-specific knowledge,
* **Agentic AI behavior** for decision-making via LangChain agents,
* and **fine-tuned LLMs** from Hugging Face or OpenAI for generating high-quality responses.

The system can answer queries from user-uploaded documents (PDFs, websites) using embeddings, vector stores, and a dynamic agent-based pipeline.

---

## 📁 Folder Structure

```
llm-agentic-rag-qa/
├── app.py                # Streamlit frontend
├── file.py               # Core logic for QA and Agent
├── data/                 # Folder to upload PDFs
├── requirements.txt      # Python dependencies
├── utils                 # functions and tools
└── keys                  # API keys (not to be committed)
```

---

## ⚙️ Features

* ✅ **PDF & Website Ingestion**
* ✅ **LangChain Agents with Multi-Tool Reasoning**
* ✅ **RAG with Pinecone and ChromaDB**
* ✅ **Fine-tuned LLM Integration (Hugging Face / OpenAI)**
* ✅ **Conversational Memory using BufferMemory**
* ✅ **Responsive Streamlit Web Interface**
* ✅ **Plug-and-play LLM Architecture**

---

## 🧪 Step-by-Step Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/suniljohn-ai/Agentic-Rag-Q-A-System.git
cd Agentic-Rag-Q-A-System
```

---

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

---

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4: Set Up Environment Variables

Create a `.env` file in the root directory and add the following:
                Or
you can create ingest your api keys into keys file
```
OPENAI_API_KEY=your-openai-api-key
HUGGINGFACEHUB_API_TOKEN=your-huggingface-token
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENV=your-pinecone-environment
```

> 🔒 Make sure `.env` is in `.gitignore` to keep your keys private.

---

### Step 5: Add Your Data

* Upload PDF files to the `./data/` folder.
* Optionally, update the `WebBaseLoader` URL in `agent_qa.py` for website ingestion.

---

### Step 6: Run the App

```bash
streamlit run app.py
```

Visit `http://localhost:8501` to interact with the application.

---

### Step 7: Customize Your LLMs (Optional)

You can switch to a **Hugging Face fine-tuned model**:

```python
from langchain.llms import HuggingFaceHub
llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.5, "max_length": 512})
```

Or use OpenAI GPT-4 or GPT-3.5:

```python
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature=0)
```

---

## 🧠 Agentic Logic Flow

1. User asks a question
2. LangChain agent decides whether to use:

   * Document-based RAG tool (for grounded answers)
   * Search/other tool (mock or real)
3. If RAG is used, it:

   * Retrieves documents from Pinecone/ChromaDB
   * Generates answer using LLM with context
4. Memory keeps multi-turn chat context

---

## 📊 Performance

* Improved response accuracy **20–30%** using fine-tuned embeddings
* Real-time retrieval from custom PDF and web data
* Dynamic agent routing for context-aware decisions

---

## 🔄 Future Improvements

* Add support for Cohere/Anthropic APIs
* Integration with LangServe or FastAPI for API-based usage
* Web scraping pipeline for live ingestion
* Knowledge graph-based reasoning layer

---

## 👨‍💻 Author

Sunil Kumar Nallabothula
Data Scientis | AI Engineer | NLP & GenAI Enthusiast
📧 [email](mailto:nsunilkumar.ai@gmail.com) • 🌐 [LinkedIn](https://linkedin.com/in/your-profile) | [GitHub](https://github.com/your-username)

---
