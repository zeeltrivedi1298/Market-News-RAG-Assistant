import logging
from typing import List, Dict
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import ConfigDict

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank-bm25 not available. Install with: pip install rank-bm25")


class HybridRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')
    
    def __init__(self, vector_retriever: BaseRetriever, documents: List[Document], alpha: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, '__dict__', {
            **self.__dict__,
            'vector_retriever': vector_retriever,
            'documents': documents,
            'alpha': alpha,
            'bm25_index': None,
            'doc_texts': None,
            'doc_map': None
        })
        if BM25_AVAILABLE and documents:
            try:
                self._initialize_bm25(documents)
            except Exception as e:
                logger.warning(f"Failed to initialize BM25, falling back to vector-only: {e}")
                self.bm25_index = None
    
    def _initialize_bm25(self, documents: List[Document]):
        if not documents:
            return
        self.doc_texts = [doc.page_content.lower().split() for doc in documents]
        self.doc_map = {i: doc for i, doc in enumerate(documents)}
        if self.doc_texts:
            try:
                self.bm25_index = BM25Okapi(self.doc_texts)
                logger.info(f"Initialized BM25 index with {len(self.doc_texts)} documents")
            except Exception as e:
                logger.warning(f"Failed to create BM25 index: {e}")
                self.bm25_index = None
    
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        if not query or not query.strip():
            return []
        
        query = query.strip()
        vector_docs = self._get_vector_docs(query, run_manager)
        
        if not self.bm25_index or not self.doc_texts:
            logger.debug("BM25 not available, using vector-only retrieval")
            return vector_docs
        
        query_tokens = query.lower().split()
        if not query_tokens:
            return vector_docs
        
        try:
            bm25_scores = self.bm25_index.get_scores(query_tokens)
            num_results = max(len(vector_docs) * 2, 20)
            top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:num_results]
            bm25_docs = [self.doc_map[i] for i in top_indices if i in self.doc_map and bm25_scores[i] > 0]
            return self._combine_results(vector_docs, bm25_docs, query_tokens, bm25_scores)
        except Exception as e:
            logger.warning(f"Error in BM25 retrieval, using vector-only: {e}")
            return vector_docs
    
    def _get_vector_docs(self, query: str, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        try:
            if hasattr(self.vector_retriever, 'get_relevant_documents'):
                return self.vector_retriever.get_relevant_documents(query)
            elif hasattr(self.vector_retriever, '_get_relevant_documents'):
                return self.vector_retriever._get_relevant_documents(query, run_manager=run_manager)
            logger.warning("Vector retriever doesn't have expected methods, using empty results")
            return []
        except Exception as e:
            logger.error(f"Error in vector retrieval: {e}")
            return []
    
    def _combine_results(self, vector_docs: List[Document], bm25_docs: List[Document], query_tokens: List[str], bm25_scores: List[float]) -> List[Document]:
        doc_scores: Dict[str, float] = {}
        
        for rank, doc in enumerate(vector_docs, 1):
            doc_id = self._get_doc_id(doc)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + (self.alpha * (1.0 / (60 + rank)))
        
        if self.bm25_index and bm25_docs and bm25_scores:
            try:
                max_bm25 = max(bm25_scores) if bm25_scores else 1.0
                min_bm25 = min(bm25_scores) if bm25_scores else 0.0
                score_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1.0
                
                for rank, doc in enumerate(bm25_docs, 1):
                    doc_id = self._get_doc_id(doc)
                    doc_index = next((i for i, mapped_doc in self.doc_map.items() if self._get_doc_id(mapped_doc) == doc_id), None)
                    if doc_index is not None and doc_index < len(bm25_scores):
                        normalized_bm25 = (bm25_scores[doc_index] - min_bm25) / score_range if score_range > 0 else 0.0
                        doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + ((1.0 - self.alpha) * (1.0 / (60 + rank)) * normalized_bm25)
            except Exception as e:
                logger.warning(f"Error combining BM25 scores: {e}")
        
        all_docs = {self._get_doc_id(doc): doc for doc in vector_docs + bm25_docs}
        scored_docs = []
        seen_ids = set()
        
        for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            if doc_id in all_docs and doc_id not in seen_ids:
                scored_docs.append(all_docs[doc_id])
                seen_ids.add(doc_id)
        
        if len(scored_docs) < len(vector_docs):
            for doc in vector_docs:
                doc_id = self._get_doc_id(doc)
                if doc_id not in seen_ids:
                    scored_docs.append(doc)
                    seen_ids.add(doc_id)
        
        return scored_docs[:len(vector_docs)] if vector_docs else scored_docs
    
    def _get_doc_id(self, doc: Document) -> str:
        if doc.metadata and 'id' in doc.metadata:
            return str(doc.metadata['id'])
        if doc.metadata and 'article_id' in doc.metadata:
            return str(doc.metadata['article_id'])
        return str(hash(doc.page_content[:100]))

