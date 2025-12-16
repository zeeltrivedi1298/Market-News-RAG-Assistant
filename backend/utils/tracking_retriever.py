from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import ConfigDict
from utils.processing_tracker import ProcessingStepsTracker

class TrackingRetrieverWrapper(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')
    
    def __init__(self, base_retriever: BaseRetriever, tracker: ProcessingStepsTracker = None, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, '__dict__', {
            **self.__dict__,
            'base_retriever': base_retriever,
            'tracker': tracker
        })
    
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        base_retriever = self.__dict__.get('base_retriever')
        tracker = self.__dict__.get('tracker')
        
        is_rerank_fetch = False
        if hasattr(base_retriever, 'search_kwargs') and isinstance(base_retriever.search_kwargs, dict):
            k_value = base_retriever.search_kwargs.get('k', 3)
            is_rerank_fetch = k_value > 10
        
        step_name = "Document Retrieval (Reranking Fetch)" if is_rerank_fetch else "Document Retrieval"
        step_desc = "Fetching documents for reranking" if is_rerank_fetch else "Retrieving relevant documents from vector store"
        
        if tracker:
            step_num = tracker.start_step(
                step_name,
                step_desc,
                {"query": query[:100] + "..." if len(query) > 100 else query}
            )
        
        try:
            if hasattr(base_retriever, 'get_relevant_documents'):
                docs = base_retriever.get_relevant_documents(query)
            elif hasattr(base_retriever, '_get_relevant_documents'):
                docs = base_retriever._get_relevant_documents(query, run_manager=run_manager)
            else:
                raise AttributeError(f"{type(base_retriever)} has no get_relevant_documents method")
            
            if tracker:
                tickers = list(set(doc.metadata.get('ticker', 'Unknown') for doc in docs if doc.metadata))
                details = {
                    "documents_retrieved": len(docs),
                    "unique_tickers": tickers[:10] if len(tickers) > 10 else tickers,
                    "retriever_type": type(base_retriever).__name__
                }
                if is_rerank_fetch:
                    details["purpose"] = "Fetching more documents for reranking"
                tracker.complete_step(step_num, details)
            
            return docs
        except Exception as e:
            if tracker:
                if 'step_num' not in locals():
                    step_num = tracker.start_step(step_name, step_desc)
                tracker.error_step(step_num, str(e))
            raise
