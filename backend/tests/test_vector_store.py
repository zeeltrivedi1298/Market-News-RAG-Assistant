import unittest
import os
import sys
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from core.vector_store import VectorStoreManager
from utils.processing_tracker import ProcessingStepsTracker


class TestVectorStoreRetrieval(unittest.TestCase):
    def setUp(self):
        self.manager = VectorStoreManager(use_contextual_retrieval=False)
    
    def test_requires_initialization_before_retriever(self):
        with self.assertRaises(ValueError) as context:
            self.manager.get_retriever()
        self.assertIn("not initialized", str(context.exception))
    
    def test_initialize_raises_error_with_empty_documents(self):
        with self.assertRaises(ValueError) as context:
            self.manager.initialize_vector_store([])
        self.assertIn("cannot be empty", str(context.exception))
    
    def test_initialize_raises_error_with_invalid_documents(self):
        invalid_docs = [Document(page_content="", metadata={})]
        
        with self.assertRaises(ValueError) as context:
            self.manager.initialize_vector_store(invalid_docs)
        self.assertIn("No valid documents", str(context.exception))
    
    def test_get_retriever_returns_retriever_when_initialized(self):
        mock_vector_store = Mock()
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        
        self.manager.vector_store = mock_vector_store
        
        retriever = self.manager.get_retriever(k=5)
        self.assertIsNotNone(retriever)
        mock_vector_store.as_retriever.assert_called_once()
    
    def test_get_retriever_validates_k(self):
        mock_vector_store = Mock()
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        self.manager.vector_store = mock_vector_store
        
        retriever = self.manager.get_retriever(k=0)
        call_args = mock_vector_store.as_retriever.call_args[1]
        self.assertEqual(call_args['search_kwargs']['k'], 1)
        
        retriever = self.manager.get_retriever(k=100)
        call_args = mock_vector_store.as_retriever.call_args[1]
        self.assertEqual(call_args['search_kwargs']['k'], 20)
    
    def test_get_retriever_with_ticker_filter(self):
        mock_vector_store = Mock()
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        self.manager.vector_store = mock_vector_store
        
        retriever = self.manager.get_retriever(k=3, ticker_filter="AAPL")
        call_args = mock_vector_store.as_retriever.call_args[1]
        
        self.assertIn("filter", call_args['search_kwargs'])
        self.assertEqual(call_args['search_kwargs']['filter']['ticker'], "AAPL")
    
    def test_get_retriever_with_tracker(self):
        mock_vector_store = Mock()
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        self.manager.vector_store = mock_vector_store
        
        tracker = ProcessingStepsTracker()
        retriever = self.manager.get_retriever(k=3, tracker=tracker)
        
        self.assertEqual(len(tracker.steps), 1)
        self.assertEqual(tracker.steps[0].name, "Building Retriever")
        self.assertEqual(tracker.steps[0].status, "completed")
    
    @patch('core.vector_store.Config')
    def test_get_retriever_with_hybrid_search_enabled(self, mock_config):
        mock_config.USE_HYBRID_SEARCH = True
        mock_config.HYBRID_SEARCH_ALPHA = 0.5
        
        mock_vector_store = Mock()
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        self.manager.vector_store = mock_vector_store
        
        chunks = [Document(page_content="Test", metadata={"id": "1"})]
        self.manager.chunks = chunks
        
        with patch('retrieval.hybrid_retriever.HybridRetriever') as mock_hybrid:
            mock_hybrid_instance = Mock()
            mock_hybrid.return_value = mock_hybrid_instance
            
            retriever = self.manager.get_retriever(k=3)
            
            mock_hybrid.assert_called_once()
            self.assertEqual(retriever, mock_hybrid_instance)
    
    @patch('core.vector_store.Config')
    def test_get_retriever_hybrid_search_fallback(self, mock_config):
        mock_config.USE_HYBRID_SEARCH = True
        
        mock_vector_store = Mock()
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        self.manager.vector_store = mock_vector_store
        
        chunks = [Document(page_content="Test", metadata={"id": "1"})]
        self.manager.chunks = chunks
        
        with patch('retrieval.hybrid_retriever.HybridRetriever') as mock_hybrid:
            mock_hybrid.side_effect = Exception("Hybrid failed")
            
            retriever = self.manager.get_retriever(k=3)
            
            self.assertEqual(retriever, mock_retriever)
    
    @patch('core.vector_store.Config')
    def test_get_retriever_vector_only_when_hybrid_disabled(self, mock_config):
        mock_config.USE_HYBRID_SEARCH = False
        
        mock_vector_store = Mock()
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        self.manager.vector_store = mock_vector_store
        
        retriever = self.manager.get_retriever(k=3)
        
        self.assertEqual(retriever, mock_retriever)


if __name__ == '__main__':
    unittest.main()
