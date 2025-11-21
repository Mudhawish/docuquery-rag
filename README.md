# DocuQuery: Advanced RAG Document Q&A System

An intelligent Retrieval-Augmented Generation (RAG) system built with IBM Granite models and LlamaIndex. Implements advanced RAG techniques learned from the IBM SkillsBuild webinar "Building Smarter RAG Systems" with Muhammad Farooq.

---

## 🚀 Features

- 📄 **Multi-format Document Support** - Process PDF, TXT, DOCX files
- 🔍 **Advanced Retrieval** - Hybrid search combining semantic and keyword matching
- 💡 **Query Expansion** - Generate multiple query variations for better retrieval
- 🎯 **LLM Reranking** - Improve answer relevance with LLM-powered ranking
- 🌐 **Streamlit Web Interface** - User-friendly interface for document Q&A
- 📚 **Source Citation** - Trace answers back to original document sources
- ⚙️ **Configurable Pipeline** - Toggle advanced features in real-time

---

## 🛠️ Tech Stack

- **LLM**: IBM Granite (via Replicate API)
- **Framework**: LlamaIndex (RAG orchestration)
- **Vector Store**: ChromaDB
- **Embeddings**: BAAI/bge-small-en-v1.5
- **UI**: Streamlit
- **Document Processing**: PyPDF, Sentence Transformers

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Replicate API account ([Sign up here](https://replicate.com/))

### Quick Start

#### Clone the repository

```bash
git clone https://github.com/yourusername/docuquery-rag.git
cd docuquery-rag
```

#### Install dependencies

```bash
pip install -r requirements.txt
```

#### Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your REPLICATE_API_TOKEN
```

#### Add your documents

```bash
mkdir -p data/sample_documents
# Add your PDF, TXT, or DOCX files to data/sample_documents/
```

#### Launch the application

```bash
streamlit run app.py
```

---

## 🔐 API Setup

1. **Get your Replicate API token:**
   - Sign up at [Replicate](https://replicate.com)
   - Go to **Account Settings**
   - Copy your API token

2. **Configure your environment:**
   ```bash
   # Edit the .env file and add:
   REPLICATE_API_TOKEN=your_actual_token_here
   ```

---

## 💡 Usage

- Start the application:
  ```bash
  streamlit run app.py
  ```
- Access the web interface:
  - Open [http://localhost:8501](http://localhost:8501) in your browser

- **Ask questions about your documents**
  - Type your question in the text input
  - Use the sidebar to toggle advanced RAG features
  - Click "Get Answer" to generate responses

**Advanced Features (toggle in sidebar):**
- Reranking: Use LLM to re-rank results for better relevance
- Query Expansion: Generate multiple query variations
- Hybrid Search: Combine semantic and keyword search

**Example Queries:**
- "What is the main topic of the documents?"
- "Can you summarize the key findings?"
- "What methodologies were used in the research?"
- "What are the limitations mentioned?"

---

## 🏗️ Project Structure

```
docuquery-rag/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variables template
├── .gitignore             # Git ignore rules
├── README.md              # Project documentation
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration settings
│   └── rag_pipeline.py    # Core RAG pipeline logic
└── data/
    └── sample_documents/  
        └── rag_tutorial.txt
```

---

## 🔧 Configuration

Modify `src/config.py` to customize the RAG pipeline:

```python
# RAG Features
ENABLE_RERANKING = False
ENABLE_QUERY_EXPANSION = False 
ENABLE_HYBRID_SEARCH = False

# Document Processing
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Models
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "meta/meta-llama-3-70b-instruct"
```

---

## 🎓 Learning Outcomes

This project demonstrates practical implementation of RAG concepts from the IBM SkillsBuild webinar:

- **RAG Architecture:** Understanding retrieval-augmented generation fundamentals
- **Advanced Retrieval:** Implementing hybrid search and query expansion
- **LLM Integration:** Working with IBM Granite models via Replicate API
- **Vector Databases:** Using ChromaDB for efficient similarity search
- **Production Deployment:** Building deployable applications with Streamlit

---

## 🐛 Troubleshooting

### Common Issues

**"REPLICATE_API_TOKEN not found"**
- Ensure you've created `.env` from `.env.example`
- Verify your token is correctly added to `.env`

**"No documents found"**
- Check that `data/sample_documents/` contains your files
- Supported formats: PDF, TXT, DOCX

**Import errors**
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

### Getting Help

- Check the [Replicate Documentation](https://replicate.com/docs)
- Review [LlamaIndex Guides](https://docs.llamaindex.ai/en/latest/)
- Open an issue on GitHub for bug reports

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- IBM SkillsBuild for the comprehensive RAG webinar
- Muhammad Farooq, for expert guidance
- IBM Granite team for the powerful language models
- LlamaIndex community for the excellent orchestration framework

---

## 📞 Support

If you encounter any issues or have questions:

- Check the troubleshooting section
- Open an issue on GitHub
- Review the documentation links provided

---

<div align="center">
  Built with curiosity using IBM Granite, LlamaIndex, and Streamlit
  <br>
  Demonstrating practical AI implementation from IBM SkillsBuild learning
</div>
