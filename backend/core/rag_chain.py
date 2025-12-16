import logging
from typing import List, Optional
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import ConfigDict
from operator import itemgetter
from config import Config
from core.vector_store import VectorStoreManager
from models.citation_models import QuotedCitations
from utils.prompts import get_rag_prompt_template, get_citations_prompt_template
from utils.formatters import format_docs_with_metadata, format_rag_result
from utils.resilience import initialize_llm_with_fallback, retry_with_backoff, _is_retryable_error
from utils.processing_tracker import ProcessingStepsTracker
from utils.tracking_retriever import TrackingRetrieverWrapper

logger = logging.getLogger(__name__)

DEFAULT_K = 3
MIN_K = 1
MAX_K = 20
MAX_QUESTION_LENGTH = 1000
MAX_RERANK_FETCH_K = 50


@retry_with_backoff(
    max_retries=Config.MAX_RETRIES,
    initial_delay=Config.INITIAL_RETRY_DELAY,
    backoff_factor=Config.BACKOFF_FACTOR,
    max_delay=Config.MAX_RETRY_DELAY
)
def _rerank_with_backoff(reranker, query: str, docs: List[Document], top_k: int) -> List[Document]:
    return reranker.rerank(query, docs, top_k=top_k)


@retry_with_backoff(
    max_retries=Config.MAX_RETRIES,
    initial_delay=Config.INITIAL_RETRY_DELAY,
    backoff_factor=Config.BACKOFF_FACTOR,
    max_delay=Config.MAX_RETRY_DELAY
)
def _invoke_llm_with_backoff(chain, inputs):
    return chain.invoke(inputs)

@retry_with_backoff(
    max_retries=Config.MAX_RETRIES,
    initial_delay=Config.INITIAL_RETRY_DELAY,
    backoff_factor=Config.BACKOFF_FACTOR,
    max_delay=Config.MAX_RETRY_DELAY
)
def _retrieve_documents(retriever: BaseRetriever, query: str, run_manager) -> List[Document]:
    if hasattr(retriever, 'get_relevant_documents'):
        return retriever.get_relevant_documents(query)
    elif hasattr(retriever, '_get_relevant_documents'):
        return retriever._get_relevant_documents(query, run_manager=run_manager)
    raise AttributeError(f"{type(retriever)} has no get_relevant_documents method")


class RerankingRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')
    
    def __init__(self, base_retriever: BaseRetriever, reranker, final_k: int, fetch_k: int, tracker: Optional[ProcessingStepsTracker] = None, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, '__dict__', {
            **self.__dict__,
            'base_retriever': base_retriever,
            'reranker': reranker,
            'final_k': final_k,
            'fetch_k': fetch_k,
            'tracker': tracker
        })
    
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        base_retriever = self.__dict__.get('base_retriever')
        reranker = self.__dict__.get('reranker')
        final_k = self.__dict__.get('final_k')
        fetch_k = self.__dict__.get('fetch_k')
        tracker = self.__dict__.get('tracker')
        
        if not base_retriever or not reranker:
            raise ValueError("base_retriever or reranker not found")
        
        try:
            docs = self._fetch_documents(base_retriever, query, fetch_k, final_k, run_manager)
            
            if len(docs) <= final_k:
                return docs
            
            if tracker:
                step_num = tracker.start_step(
                    "Document Reranking",
                    f"Reranking {len(docs)} documents to select top {final_k}",
                    {"documents_fetched": len(docs), "final_k": final_k}
                )
            
            reranked_docs = _rerank_with_backoff(reranker, query, docs, final_k)
            logger.debug(f"Reranked {len(docs)} documents to top {len(reranked_docs)}")
            
            if tracker:
                tracker.complete_step(step_num, {
                    "documents_reranked": len(reranked_docs),
                    "documents_before_rerank": len(docs)
                })
            
            return reranked_docs
            
        except Exception as e:
            logger.warning(f"Error during reranking, returning original results: {e}")
            if tracker:
                step_num = tracker.start_step("Document Reranking", "Reranking documents")
                tracker.error_step(step_num, str(e))
            return self._fallback_retrieval(base_retriever, query, final_k, run_manager)
    
    def _fetch_documents(self, retriever: BaseRetriever, query: str, fetch_k: int, 
                        final_k: int, run_manager) -> List[Document]:
        original_k = None
        
        if hasattr(retriever, 'search_kwargs') and isinstance(retriever.search_kwargs, dict):
            original_k = retriever.search_kwargs.get('k', final_k)
            retriever.search_kwargs['k'] = fetch_k
        
        try:
            docs = _retrieve_documents(retriever, query, run_manager)
        finally:
            if original_k is not None and hasattr(retriever, 'search_kwargs'):
                retriever.search_kwargs['k'] = original_k
        
        return docs
    
    def _fallback_retrieval(self, retriever: BaseRetriever, query: str, 
                           final_k: int, run_manager) -> List[Document]:
        try:
            docs = _retrieve_documents(retriever, query, run_manager)
            return docs[:final_k]
        except Exception as fallback_error:
            logger.error(f"Fallback retrieval also failed: {fallback_error}")
            return []

class RAGChain:
    def __init__(self, vector_store_manager: VectorStoreManager):
        if vector_store_manager is None:
            raise ValueError("vector_store_manager cannot be None")
        self.vector_store_manager = vector_store_manager
        self.llm = self._initialize_llm()
        self.structured_llm = None
        if self.llm:
            try:
                self.structured_llm = self.llm.with_structured_output(QuotedCitations, method='function_calling')
            except Exception as e:
                logger.warning(f"Failed to create structured LLM: {str(e)}")
    
    def _initialize_llm(self):
        llm = initialize_llm_with_fallback(temperature=0)
        if not llm:
            raise ValueError("OPENAI_API_KEY is required and both primary and fallback LLMs failed")
        return llm
    
    def create_rag_chain(self, ticker_filter: str = None, k: int = DEFAULT_K, tracker: Optional[ProcessingStepsTracker] = None):
        if not self.llm:
            raise ValueError("LLM not initialized. Check OpenAI API key configuration.")
        
        k = self._validate_k(k)
        ticker_filter = self._normalize_ticker_filter(ticker_filter)
        retriever = self._build_retriever(k, ticker_filter, tracker)
        return self._assemble_rag_pipeline(retriever, tracker)
    
    def _validate_k(self, k: int) -> int:
        if not isinstance(k, int):
            k = DEFAULT_K
        return max(MIN_K, min(MAX_K, k))
    
    def _normalize_ticker_filter(self, ticker_filter: str) -> str:
        if ticker_filter:
            return ticker_filter.strip()
        return None
    
    def _build_retriever(self, k: int, ticker_filter: str, tracker: Optional[ProcessingStepsTracker] = None) -> BaseRetriever:
        retriever = self.vector_store_manager.get_retriever(k=k, ticker_filter=ticker_filter, tracker=tracker)
        
        if tracker:
            retriever = TrackingRetrieverWrapper(retriever, tracker)
        
        if k > 5:
            retriever = self._wrap_with_reranker(retriever, k, tracker)
        return retriever
    
    def _assemble_rag_pipeline(self, retriever: BaseRetriever, tracker: Optional[ProcessingStepsTracker] = None):
        return (
            {"context": retriever, "question": RunnablePassthrough()}
            | RunnablePassthrough.assign(answer=self._create_answer_chain(tracker))
            | RunnablePassthrough.assign(citations=self._create_citations_chain(tracker))
        )
    
    def _wrap_with_reranker(self, retriever: BaseRetriever, final_k: int, tracker: Optional[ProcessingStepsTracker] = None) -> BaseRetriever:
        from retrieval.reranker import DocumentReranker
        
        reranker = DocumentReranker()
        fetch_k = max(final_k, min(final_k * Config.RERANK_TOP_K_MULTIPLIER, MAX_RERANK_FETCH_K))
        return RerankingRetriever(retriever, reranker, final_k, fetch_k, tracker)
    
    def _create_answer_chain(self, tracker: Optional[ProcessingStepsTracker] = None):
        rag_prompt = get_rag_prompt_template()
        answer_chain = (
            {
                "context": (itemgetter('context') | RunnableLambda(format_docs_with_metadata)),
                "question": itemgetter("question")
            }
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )
        
        if tracker:
            def track_answer_generation(inputs):
                step_num = tracker.start_step(
                    "Answer Generation",
                    "Generating answer using retrieved documents and LLM",
                    {"question": inputs.get("question", "")[:100] if isinstance(inputs, dict) else ""}
                )
                
                try:
                    context = inputs.get("context", []) if isinstance(inputs, dict) else []
                    if not context or (isinstance(context, list) and len(context) == 0):
                        error_msg = "No documents retrieved. Cannot generate answer without context."
                        tracker.error_step(step_num, error_msg)
                        raise ValueError(error_msg)
                    
                    answer = _invoke_llm_with_backoff(answer_chain, inputs)
                    
                    if not answer or not answer.strip():
                        error_msg = "LLM returned empty answer."
                        tracker.error_step(step_num, error_msg)
                        raise ValueError(error_msg)
                    
                    tracker.complete_step(step_num, {
                        "answer_length": len(answer),
                        "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
                        "documents_used": len(context) if isinstance(context, list) else 0
                    })
                    
                    return answer
                except Exception as e:
                    tracker.error_step(step_num, str(e))
                    raise
            
            return RunnableLambda(track_answer_generation)
        
        return answer_chain
    
    def _create_citations_chain(self, tracker: Optional[ProcessingStepsTracker] = None):
        if not self.structured_llm:
            return RunnableLambda(lambda x: QuotedCitations(citations=[]))
        
        cite_prompt = get_citations_prompt_template()
        citations_chain = (
            {
                "context": (itemgetter('context') | RunnableLambda(format_docs_with_metadata)),
                "question": itemgetter("question"),
                "answer": itemgetter("answer")
            }
            | cite_prompt
            | self.structured_llm
        )
        
        if tracker:
            def track_citation_extraction(inputs):
                step_num = tracker.start_step(
                    "Citation Extraction",
                    "Extracting citations from the generated answer",
                    {}
                )
                
                try:
                    answer = inputs.get("answer", "") if isinstance(inputs, dict) else ""
                    if not answer or not answer.strip():
                        tracker.complete_step(step_num, {
                            "citations_count": 0,
                            "reason": "No answer to extract citations from"
                        })
                        return QuotedCitations(citations=[])
                    
                    citations = _invoke_llm_with_backoff(citations_chain, inputs)
                    
                    citations_list = citations.citations if hasattr(citations, 'citations') else []
                    tracker.complete_step(step_num, {
                        "citations_count": len(citations_list)
                    })
                    
                    return citations
                except Exception as e:
                    logger.warning(f"Citation extraction failed: {e}. Returning empty citations.")
                    tracker.error_step(step_num, str(e))
                    return QuotedCitations(citations=[])
            
            return RunnableLambda(track_citation_extraction)
        
        return citations_chain
    
    def query(self, question: str, ticker_filter: str = None, k: int = DEFAULT_K, tracker: Optional[ProcessingStepsTracker] = None) -> dict:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        question = self._normalize_question(question)
        chain = self.create_rag_chain(ticker_filter=ticker_filter, k=k, tracker=tracker)
        result = _invoke_llm_with_backoff(chain, question)
        return format_rag_result(result)
    
    def _normalize_question(self, question: str) -> str:
        question = question.strip()
        if len(question) > MAX_QUESTION_LENGTH:
            question = question[:MAX_QUESTION_LENGTH]
            logger.warning(f"Question truncated to {MAX_QUESTION_LENGTH} characters")
        return question
    
    def query_for_evaluation(self, question: str, ticker_filter: str = None, k: int = DEFAULT_K) -> dict:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        question = self._normalize_question(question)
        chain = self.create_rag_chain(ticker_filter=ticker_filter, k=k)
        return _invoke_llm_with_backoff(chain, question)
