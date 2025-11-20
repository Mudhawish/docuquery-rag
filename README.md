# DocuQuery: Advanced RAG Document Q&A System

An intelligent Retrieval-Augmented Generation (RAG) system built with IBM Granite models and LlamaIndex. Implements advanced RAG techniques learned from the IBM SkillsBuild webinar "**Building Smarter RAG Systems**" with Muhammad Farooq.

---

## 🚀 Features

- **Multi-format Document Support:** Process PDF, TXT, DOCX files
- **Advanced Retrieval:** Hybrid search combining semantic and keyword matching
- **Query Expansion:** Generate multiple query variations for better retrieval
- **LLM Reranking:** Improve answer relevance with LLM-powered ranking
- **Streamlit Web Interface:** User-friendly interface for document Q&A
- **Source Citation:** Trace answers back to original document sources

---

## 🛠️ Tech Stack

- **LLM:** IBM Granite (via Replicate API)
- **Framework:** LlamaIndex (RAG orchestration)
- **Vector Store:** ChromaDB
- **Embeddings:** BAAI/bge-small-en-v1.5
- **UI:** Streamlit

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- Replicate API token

### Quick Start

1. **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/docuquery-rag.git
    cd docuquery-rag
    ```
2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3. **Set up environment variables**
    ```bash
    cp .env.example .env
    # Add your REPLICATE_API_TOKEN to .env
    ```
4. **Add sample documents**
    ```bash
    mkdir -p data/sample_documents
    # Add your PDF, TXT, or DOCX files to data/sample_documents/
    ```
5. **Launch the application**
    ```bash
    streamlit run app.py
    ```

---

## 💡 Usage

- **Start the application:**  
  `streamlit run app.py`
- **Access the web interface:**  
  Open [http://localhost:8501](http://localhost:8501) in your browser
- **Ask questions:**  
  Type questions about your documents
- **Toggle features:**  
  Use the sidebar to enable advanced RAG features

---

## 🏗️ Project Structure

```
docuquery-rag/
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variables template
├── src/
│   ├── config.py         # Configuration settings
│   └── rag_pipeline.py   # Core RAG pipeline logic
└── data/
    └── sample_documents/ # Place your documents here
```

---

## 🔧 Configuration

Modify `src/config.py` to customize RAG features and model settings.

---

## 🎓 Learning Outcomes

This project demonstrates practical implementation of RAG concepts from the IBM SkillsBuild webinar:

- **RAG Architecture:** Understanding retrieval-augmented generation fundamentals
- **Advanced Retrieval:** Implementing hybrid search and query expansion
- **LLM Integration:** Working with IBM Granite models
- **Vector Databases:** Using ChromaDB for similarity search

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments

- IBM SkillsBuild for the comprehensive RAG webinar
- Muhammad Farooq for expert guidance
- IBM Granite team for the language models
