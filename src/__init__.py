"""
DocuQuery RAG System
Advanced Retrieval-Augmented Generation with IBM Granite and LlamaIndex
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Advanced RAG Document Q&A System"

# Import key components for easy access
from .config import Config
from .rag_pipeline import RAGPipeline

__all__ = ["Config", "RAGPipeline"]
