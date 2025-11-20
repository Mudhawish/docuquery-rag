import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # RAG Features
    ENABLE_RERANKING = False
    ENABLE_QUERY_EXPANSION = False 
    ENABLE_HYBRID_SEARCH = False
    
    # Document Processing
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    MAX_DOCUMENT_SIZE_MB = 10
    
    # Models
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    LLM_MODEL = "meta/meta-llama-3-70b-instruct"
    
    # Vector Store
    VECTOR_STORE_PATH = "./chroma_db"
    COLLECTION_NAME = "rag_docs"
    
    # API Keys (from environment)
    REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
    
    @classmethod
    def validate_config(cls):
        if not cls.REPLICATE_API_TOKEN:
            raise ValueError("REPLICATE_API_TOKEN not found in environment variables")
