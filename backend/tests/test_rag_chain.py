import unittest
import os
import sys
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from core.rag_chain import RAGChain, RerankingRetriever, DEFAULT_K, MIN_K, MAX_K
from core.vector_store import VectorStoreManager
from utils.processing_tracker import ProcessingStepsTracker


class TestRAGChain(unittest.TestCase):
    def setUp(self):
        self.mock_vector_store = Mock(spec=VectorStoreManager)
        self.mock_retriever = Mock()
        self.mock_vector_store.get_retriever.return_value = self.mock_retriever
    
    @patch('core.rag_chain.initialize_llm_with_fallback')
    def test_init_with_valid_vector_store(self, mock_init_llm):
        mock_llm = Mock()
        mock_init_llm.return_value = mock_llm
        
        rag_chain = RAGChain(self.mock_vector_store)
        
        self.assertEqual(rag_chain.vector_store_manager, self.mock_vector_store)
        self.assertEqual(rag_chain.llm, mock_llm)
        mock_init_llm.assert_called_once()
    
    def test_init_with_none_vector_store(self):
        with self.assertRaises(ValueError) as context:
            RAGChain(None)
        
        self.assertIn("cannot be None", str(context.exception))
    
    @patch('core.rag_chain.initialize_llm_with_fallback')
    def test_init_creates_structured_llm(self, mock_init_llm):
        mock_llm = Mock()
        mock_structured_llm = Mock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_init_llm.return_value = mock_llm
        
        rag_chain = RAGChain(self.mock_vector_store)
        
        self.assertEqual(rag_chain.structured_llm, mock_structured_llm)
    
    @patch('core.rag_chain.initialize_llm_with_fallback')
    def test_init_handles_structured_llm_failure(self, mock_init_llm):
        mock_llm = Mock()
        mock_llm.with_structured_output.side_effect = Exception("Failed")
        mock_init_llm.return_value = mock_llm
        
        rag_chain = RAGChain(self.mock_vector_store)
        
        self.assertIsNone(rag_chain.structured_llm)
    
    @patch('core.rag_chain.initialize_llm_with_fallback')
    def test_validate_k(self, mock_init_llm):
        mock_init_llm.return_value = Mock()
        rag_chain = RAGChain(self.mock_vector_store)
        
        self.assertEqual(rag_chain._validate_k(5), 5)
        self.assertEqual(rag_chain._validate_k(1), 1)
        self.assertEqual(rag_chain._validate_k(20), 20)
        self.assertEqual(rag_chain._validate_k(0), MIN_K)
        self.assertEqual(rag_chain._validate_k(100), MAX_K)
        self.assertEqual(rag_chain._validate_k("invalid"), DEFAULT_K)
    
    @patch('core.rag_chain.initialize_llm_with_fallback')
    def test_normalize_ticker_filter(self, mock_init_llm):
        mock_init_llm.return_value = Mock()
        rag_chain = RAGChain(self.mock_vector_store)
        
        self.assertEqual(rag_chain._normalize_ticker_filter("AAPL"), "AAPL")
        self.assertEqual(rag_chain._normalize_ticker_filter("  MSFT  "), "MSFT")
        self.assertIsNone(rag_chain._normalize_ticker_filter(None))
        self.assertIsNone(rag_chain._normalize_ticker_filter(""))
    
    @patch('core.rag_chain.initialize_llm_with_fallback')
    def test_create_rag_chain_without_llm(self, mock_init_llm):
        mock_init_llm.return_value = None
        
        with self.assertRaises(ValueError):
            RAGChain(self.mock_vector_store)
    
    @patch('core.rag_chain.initialize_llm_with_fallback')
    @patch('core.rag_chain.TrackingRetrieverWrapper')
    def test_build_retriever_with_tracker(self, mock_tracker_wrapper, mock_init_llm):
        mock_init_llm.return_value = Mock()
        mock_wrapped_retriever = Mock()
        mock_tracker_wrapper.return_value = mock_wrapped_retriever
        
        rag_chain = RAGChain(self.mock_vector_store)
        tracker = ProcessingStepsTracker()
        
        retriever = rag_chain._build_retriever(k=3, ticker_filter="AAPL", tracker=tracker)
        
        mock_tracker_wrapper.assert_called_once()
        self.mock_vector_store.get_retriever.assert_called_once_with(k=3, ticker_filter="AAPL", tracker=tracker)
    
    @patch('core.rag_chain.initialize_llm_with_fallback')
    @patch('core.rag_chain.RerankingRetriever')
    @patch('retrieval.reranker.DocumentReranker')
    def test_build_retriever_with_reranking(self, mock_reranker, mock_reranking_retriever, mock_init_llm):
        mock_init_llm.return_value = Mock()
        mock_reranker_instance = Mock()
        mock_reranker.return_value = mock_reranker_instance
        mock_reranking_retriever.return_value = Mock()
        
        rag_chain = RAGChain(self.mock_vector_store)
        
        retriever = rag_chain._build_retriever(k=10, ticker_filter=None, tracker=None)
        
        mock_reranking_retriever.assert_called_once()
    
    @patch('core.rag_chain.initialize_llm_with_fallback')
    def test_build_retriever_without_reranking(self, mock_init_llm):
        mock_init_llm.return_value = Mock()
        rag_chain = RAGChain(self.mock_vector_store)
        
        retriever = rag_chain._build_retriever(k=3, ticker_filter=None, tracker=None)
        
        self.assertEqual(retriever, self.mock_retriever)
    
    @patch('core.rag_chain.initialize_llm_with_fallback')
    def test_create_rag_chain(self, mock_init_llm):
        mock_init_llm.return_value = Mock()
        rag_chain = RAGChain(self.mock_vector_store)
        
        chain = rag_chain.create_rag_chain(ticker_filter="AAPL", k=5)
        
        self.assertIsNotNone(chain)
        self.mock_vector_store.get_retriever.assert_called_once()
    
    @patch('core.rag_chain.initialize_llm_with_fallback')
    def test_normalize_question(self, mock_init_llm):
        mock_init_llm.return_value = Mock()
        rag_chain = RAGChain(self.mock_vector_store)
        
        self.assertEqual(rag_chain._normalize_question("  Test Question  "), "Test Question")
        self.assertEqual(rag_chain._normalize_question("Test"), "Test")
    
    @patch('core.rag_chain.initialize_llm_with_fallback')
    def test_normalize_question_truncates_long_question(self, mock_init_llm):
        mock_init_llm.return_value = Mock()
        rag_chain = RAGChain(self.mock_vector_store)
        
        long_question = "A" * 2000
        result = rag_chain._normalize_question(long_question)
        
        self.assertLessEqual(len(result), 1000)
    
    @patch('core.rag_chain.initialize_llm_with_fallback')
    def test_query_with_empty_question(self, mock_init_llm):
        mock_init_llm.return_value = Mock()
        rag_chain = RAGChain(self.mock_vector_store)
        
        with self.assertRaises(ValueError) as context:
            rag_chain.query("")
        
        self.assertIn("cannot be empty", str(context.exception))
    
    @patch('core.rag_chain.initialize_llm_with_fallback')
    def test_query_with_whitespace_only(self, mock_init_llm):
        mock_init_llm.return_value = Mock()
        rag_chain = RAGChain(self.mock_vector_store)
        
        with self.assertRaises(ValueError):
            rag_chain.query("   ")


class TestRerankingRetriever(unittest.TestCase):
    def setUp(self):
        self.mock_base_retriever = Mock()
        self.mock_reranker = Mock()
        self.mock_run_manager = Mock()
    
    def test_reranking_retriever_initialization(self):
        retriever = RerankingRetriever(
            base_retriever=self.mock_base_retriever,
            reranker=self.mock_reranker,
            final_k=5,
            fetch_k=10
        )
        
        self.assertEqual(retriever.__dict__.get('final_k'), 5)
        self.assertEqual(retriever.__dict__.get('fetch_k'), 10)
    
    def test_get_relevant_documents_without_base_retriever(self):
        retriever = RerankingRetriever(
            base_retriever=None,
            reranker=self.mock_reranker,
            final_k=5,
            fetch_k=10
        )
        
        with self.assertRaises(ValueError):
            retriever._get_relevant_documents("query", run_manager=self.mock_run_manager)
    
    def test_get_relevant_documents_skips_reranking_when_not_enough_docs(self):
        docs = [Document(page_content=f"Doc {i}") for i in range(3)]
        self.mock_base_retriever.get_relevant_documents.return_value = docs
        
        retriever = RerankingRetriever(
            base_retriever=self.mock_base_retriever,
            reranker=self.mock_reranker,
            final_k=5,
            fetch_k=10
        )
        
        result = retriever._get_relevant_documents("query", run_manager=self.mock_run_manager)
        
        self.assertEqual(len(result), 3)
        self.mock_reranker.rerank.assert_not_called()
    
    def test_get_relevant_documents_performs_reranking(self):
        docs = [Document(page_content=f"Doc {i}") for i in range(10)]
        reranked_docs = docs[:5]
        
        self.mock_base_retriever.get_relevant_documents.return_value = docs
        self.mock_reranker.rerank.return_value = reranked_docs
        
        retriever = RerankingRetriever(
            base_retriever=self.mock_base_retriever,
            reranker=self.mock_reranker,
            final_k=5,
            fetch_k=10,
            tracker=None
        )
        
        result = retriever._get_relevant_documents("query", run_manager=self.mock_run_manager)
        
        self.assertEqual(len(result), 5)
        self.mock_reranker.rerank.assert_called_once()


if __name__ == '__main__':
    unittest.main()

