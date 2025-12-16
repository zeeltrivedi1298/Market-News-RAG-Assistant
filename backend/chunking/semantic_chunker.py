import logging
from typing import List
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

try:
    from langchain_experimental.text_splitter import SemanticChunker as LangChainSemanticChunker
    LANGCHAIN_SEMANTIC_AVAILABLE = True
except ImportError:
    LANGCHAIN_SEMANTIC_AVAILABLE = False


class SemanticChunker:
    def __init__(self, embeddings, similarity_threshold: float = 0.5):
        if not LANGCHAIN_SEMANTIC_AVAILABLE:
            raise ImportError(
                "langchain-experimental is required for semantic chunking. "
                "Install with: pip install langchain-experimental"
            )
        if not embeddings:
            raise ValueError("embeddings are required for semantic chunking")
        
        try:
            self.chunker = LangChainSemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=similarity_threshold
            )
            logger.info(f"Initialized LangChain SemanticChunker with threshold: {similarity_threshold}")
        except Exception as e:
            logger.error(f"Failed to initialize LangChain SemanticChunker: {e}")
            raise
    
    def chunk_by_semantics(self, documents: List[Document]) -> List[Document]:
        try:
            all_chunks = []
            for doc in documents:
                chunks = self.chunker.create_documents([doc.page_content])
                for chunk in chunks:
                    chunk.metadata = doc.metadata.copy() if doc.metadata else {}
                all_chunks.extend(chunks)
            return all_chunks if all_chunks else documents
        except Exception as e:
            logger.error(f"Error in semantic chunking: {e}")
            return documents

