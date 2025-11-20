import streamlit as st
from src.rag_pipeline import RAGPipeline
from src.config import Config

st.title("DocuQuery: Advanced Document Q&A System")
st.markdown("Upload docs, ask questions, and toggle advanced RAG features!")

@st.cache_resource
def load_rag():
    return RAGPipeline()

rag = load_rag()

st.sidebar.header("Options")
enable_reranking = st.sidebar.checkbox("Enable Reranking", value=Config().ENABLE_RERANKING)
enable_expansion = st.sidebar.checkbox("Enable Query Expansion", value=Config().ENABLE_QUERY_EXPANSION)
enable_hybrid = st.sidebar.checkbox("Enable Hybrid Search", value=Config().ENABLE_HYBRID_SEARCH)

question = st.text_input("Ask a question about the documents")

if st.button("Get Answer"):
    if question.strip():
        options = {
            "reranking": enable_reranking,
            "query_expansion": enable_expansion,
            "hybrid_search": enable_hybrid
        }
        with st.spinner("Processing..."):
            try:
                result = rag.query(question, options)
                st.subheader("Answer:")
                st.write(result["answer"])
                
                if result["sources"]:
                    st.subheader("Sources:")
                    for i, source in enumerate(result["sources"], 1):
                        st.markdown(f"Source {i} (Score: {source['score']:.2f}): {source['text'][:300]}...")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.error("Please enter a question.")

st.markdown("---")
st.markdown("Powered by RAG with IBM Granite, LlamaIndex, ChromaDB")
