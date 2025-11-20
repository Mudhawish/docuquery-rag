from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.llms.replicate import Replicate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import LLMRerank
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

load_dotenv()

@dataclass
class RAGConfig:
    """Configuration class for RAG pipeline"""
    document_path: str = "data/sample_documents"
    chroma_db_path: str = "./chroma_db"
    collection_name: str = "rag_docs"
    chunk_size: int = 512
    chunk_overlap: int = 50
    similarity_top_k: int = 5  # Increased for reranking
    rerank_top_n: int = 3
    llm_model: str = "meta/meta-llama-3-70b-instruct"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    temperature: float = 0.1
    top_p: float = 0.9
    max_new_tokens: int = 512

class RAGPipeline:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.llm = None
        self.embed_model = None
        self.index = None
        self.vector_store = None
        self.storage_context = None
        self.initialize_components()
        
    def initialize_components(self):
        """Initialize LLM, embedding model, and vector index"""
        try:
            # Validate API token first
            api_token = os.getenv("REPLICATE_API_TOKEN")
            if not api_token:
                raise ValueError(
                    "REPLICATE_API_TOKEN not found. "
                    "Please add it to your .env file. "
                    "Get it from: https://replicate.com/account"
                )
            
            # Initialize LLM
            self.llm = Replicate(
                model=self.config.llm_model,
                temperature=self.config.temperature,
                additional_kwargs={
                    "top_p": self.config.top_p, 
                    "max_new_tokens": self.config.max_new_tokens
                }
            )
            
            # Initialize embedding model
            self.embed_model = HuggingFaceEmbedding(
                model_name=self.config.embed_model
            )
            
            # Configure settings
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            Settings.chunk_size = self.config.chunk_size
            Settings.chunk_overlap = self.config.chunk_overlap
            
            # Load documents and create index
            self.load_documents()
            
        except Exception as e:
            print(f"Error initializing components: {e}")
            raise
    
    def load_documents(self):
        """Load documents with enhanced processing and create vector index"""
        try:
            # Check if documents exist
            if not os.path.exists(self.config.document_path):
                raise FileNotFoundError(f"Document path not found: {self.config.document_path}")
                
            # Load documents with metadata
            documents = SimpleDirectoryReader(
                self.config.document_path,
                file_metadata=lambda filename: {"file_name": os.path.basename(filename)}
            ).load_data()
            
            if not documents:
                raise ValueError("No documents found in the specified path")
            
            # Create node parser with custom settings
            node_parser = SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separator=" ",
                paragraph_separator="\n\n"
            )
            
            # Create vector store
            db = chromadb.PersistentClient(path=self.config.chroma_db_path)
            chroma_collection = db.get_or_create_collection(self.config.collection_name)
            self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            # Create index with node parser
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                embed_model=self.embed_model,
                transformations=[node_parser]
            )
            
            print(f"✅ Successfully loaded {len(documents)} documents into ChromaDB")
            
        except Exception as e:
            print(f"Error loading documents: {e}")
            raise
    
    def create_enhanced_query_engine(self):
        """Create query engine with reranking for better results"""
        # Initialize reranker
        rerank = LLMRerank(
            choice_batch_size=5,
            top_n=self.config.rerank_top_n,
            llm=self.llm
        )
        
        query_engine = self.index.as_query_engine(
            llm=self.llm,
            similarity_top_k=self.config.similarity_top_k,
            node_postprocessors=[rerank],
            vector_store_query_mode="default"
        )
        return query_engine
    
    def query(self, question: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the RAG pipeline with enhanced response formatting"""
        if options is None:
            options = {}
            
        try:
            # Use enhanced query engine with reranking
            query_engine = self.create_enhanced_query_engine()
            
            # Execute query
            response = query_engine.query(question)
            
            # Format response
            result = {
                "answer": str(response),
                "sources": [],
                "metadata": {
                    "question": question,
                    "sources_count": 0
                }
            }
            
            # Add source information if available
            if hasattr(response, 'source_nodes'):
                result["metadata"]["sources_count"] = len(response.source_nodes)
                
                for i, node in enumerate(response.source_nodes):
                    source_info = {
                        "source_id": i,
                        "text": self._truncate_text(node.node.text, 500),
                        "score": float(node.score) if node.score else 0.0,
                        "metadata": node.node.metadata or {}
                    }
                    result["sources"].append(source_info)
            
            return result
            
        except Exception as e:
            print(f"Error during query: {e}")
            return {
                "answer": f"Error processing your question: {str(e)}",
                "sources": [],
                "metadata": {
                    "question": question,
                    "error": str(e),
                    "sources_count": 0
                }
            }
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Helper method to truncate text for display"""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector index"""
        try:
            if not self.index:
                return {"error": "Index not initialized"}
                
            # Get collection stats from ChromaDB
            db = chromadb.PersistentClient(path=self.config.chroma_db_path)
            collection = db.get_collection(self.config.collection_name)
            
            stats = {
                "collection_name": self.config.collection_name,
                "total_documents": collection.count(),
                "embedding_model": self.config.embed_model,
                "llm_model": self.config.llm_model,
                "chunk_size": self.config.chunk_size,
                "similarity_top_k": self.config.similarity_top_k
            }
            
            return stats
            
        except Exception as e:
            return {"error": f"Could not retrieve stats: {str(e)}"}
    
    def clear_database(self):
        """Clear the vector database"""
        try:
            db = chromadb.PersistentClient(path=self.config.chroma_db_path)
            db.delete_collection(self.config.collection_name)
            print("✅ Database cleared successfully")
        except Exception as e:
            print(f"Error clearing database: {e}")

# Example usage
if __name__ == "__main__":
    # Custom configuration
    config = RAGConfig(
        document_path="data/documents",
        chroma_db_path="./my_chroma_db",
        chunk_size=1024,
        similarity_top_k=5
    )
    
    # Initialize pipeline
    rag = RAGPipeline(config)
    
    # Get statistics
    stats = rag.get_index_stats()
    print("Index Stats:", stats)
    
    # Query example
    result = rag.query("What is the main topic of the documents?")
    print("Answer:", result["answer"])
    print("Sources:", len(result["sources"]))
    
    # Query with options
    result2 = rag.query(
        "Explain the key concepts discussed",
        options={"similarity_top_k": 3}
    )
    print("Answer 2:", result2["answer"])
