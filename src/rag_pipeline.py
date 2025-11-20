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

load_dotenv()

class RAGPipeline:
    def __init__(self, document_path="data/sample_documents"):
        self.document_path = document_path
        self.llm = None
        self.embed_model = None
        self.index = None
        self.initialize_components()
        
    def initialize_components(self):
        """Initialize LLM, embedding model, and vector index"""
        try:
            # Initialize LLM
            self.llm = Replicate(
                model="meta/meta-llama-3-70b-instruct",
                temperature=0.1,
                additional_kwargs={"top_p": 0.9, "max_new_tokens": 512}
            )
            
            # Initialize embedding model
            self.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5"
            )
            
            # Configure settings
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            Settings.chunk_size = 512
            Settings.chunk_overlap = 50
            
            # Load documents and create index
            self.load_documents()
            
        except Exception as e:
            print(f"Error initializing components: {e}")
            raise
    
    def load_documents(self):
        """Load documents and create vector index"""
        try:
            # Check if documents exist
            if not os.path.exists(self.document_path):
                raise FileNotFoundError(f"Document path not found: {self.document_path}")
                
            # Load documents
            documents = SimpleDirectoryReader(self.document_path).load_data()
            
            if not documents:
                raise ValueError("No documents found in the specified path")
            
            # Create vector store
            db = chromadb.PersistentClient(path="./chroma_db")
            chroma_collection = db.get_or_create_collection("rag_docs")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create index
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=self.embed_model
            )
            
            print(f"✅ Successfully loaded {len(documents)} documents")
            
        except Exception as e:
            print(f"Error loading documents: {e}")
            raise
    
    def query(self, question, options=None):
        """Query the RAG pipeline"""
        if options is None:
            options = {}
            
        try:
            # Create query engine
            query_engine = self.index.as_query_engine(
                llm=self.llm,
                similarity_top_k=3,
                vector_store_query_mode="default"
            )
            
            # Execute query
            response = query_engine.query(question)
            
            # Format response
            result = {
                "answer": str(response),
                "sources": []
            }
            
            # Add source information if available
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    result["sources"].append({
                        "text": node.node.text[:500],  # Limit text length
                        "score": float(node.score) if node.score else 0.0
                    })
            
            return result
            
        except Exception as e:
            print(f"Error during query: {e}")
            return {
                "answer": f"Error processing your question: {str(e)}",
                "sources": []
            }