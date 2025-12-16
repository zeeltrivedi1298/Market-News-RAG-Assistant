import logging
from typing import List, Optional
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
except ImportError:
    FLASHRANK_AVAILABLE = False
    logger.warning("flashrank not available. Install with: pip install flashrank")


class DocumentReranker:
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2", max_length: int = 512):
        if not FLASHRANK_AVAILABLE:
            raise ImportError(
                "flashrank is required but not installed. "
                "Please install it with: pip install flashrank"
            )
        try:
            self.ranker = Ranker(model_name=model_name, max_length=max_length)
            logger.info(f"Initialized flashrank reranker with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize flashrank reranker: {e}")
            raise RuntimeError(f"Failed to initialize flashrank reranker: {e}") from e
    
    def rerank(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> List[Document]:
        if not self.ranker or not documents or not query or not query.strip():
            return documents[:top_k] if top_k else documents
        
        try:
            MAX_RERANK_TEXT_LENGTH = 1000
            passages = [
                {
                    "id": str(i),
                    "text": doc.page_content[:MAX_RERANK_TEXT_LENGTH] + "..." 
                        if len(doc.page_content) > MAX_RERANK_TEXT_LENGTH 
                        else doc.page_content
                }
                for i, doc in enumerate(documents)
            ]
            
            if not passages:
                return documents[:top_k] if top_k else documents
            
            reranked_results = self.ranker.rerank(RerankRequest(query=query.strip(), passages=passages))
            doc_map = {str(i): doc for i, doc in enumerate(documents)}
            
            reranked_docs = []
            for result in reranked_results:
                doc_id = result.get("id", "")
                if doc_id in doc_map:
                    original_doc = doc_map[doc_id]
                    new_metadata = (original_doc.metadata.copy() if original_doc.metadata else {})
                    new_metadata["rerank_score"] = result.get("score", 0.0)
                    reranked_docs.append(Document(
                        page_content=original_doc.page_content,
                        metadata=new_metadata
                    ))
            
            if len(reranked_docs) < len(documents):
                reranked_ids = {result.get("id", "") for result in reranked_results}
                reranked_docs.extend(doc for i, doc in enumerate(documents) if str(i) not in reranked_ids)
            
            return reranked_docs[:top_k] if top_k else reranked_docs
        except Exception as e:
            logger.error(f"Error during reranking: {e}", exc_info=True)
            return documents[:top_k] if top_k else documents

