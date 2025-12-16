import unittest
import os
import sys
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_loader import load_stock_news
from core.vector_store import VectorStoreManager
from core.rag_chain import RAGChain
from langchain_core.documents import Document
from utils.processing_tracker import ProcessingStepsTracker


class TestRAGIntegration(unittest.TestCase):
    def setUp(self):
        self.test_data = {
            "AAPL": [
                {
                    "title": "Apple Announces New Product",
                    "link": "https://example.com/apple",
                    "full_text": "Apple Inc. announced a new iPhone model with advanced AI features. The company expects strong sales in the upcoming quarter."
                }
            ],
            "MSFT": [
                {
                    "title": "Microsoft Cloud Growth",
                    "link": "https://example.com/microsoft",
                    "full_text": "Microsoft reported strong cloud revenue growth. Azure services saw significant adoption across enterprises."
                }
            ]
        }
        
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.test_data, self.temp_file)
        self.temp_file.close()
    
    def tearDown(self):
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    @patch('core.rag_chain.initialize_llm_with_fallback')
    def test_rag_chain_with_mocked_llm(self, mock_init_llm):
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value="Mocked answer")
        mock_structured_llm = Mock()
        mock_structured_llm.invoke = Mock(return_value=Mock(citations=[]))
        mock_llm.with_structured_output = Mock(return_value=mock_structured_llm)
        mock_init_llm.return_value = mock_llm
        
        documents = load_stock_news(self.temp_file.name)
        
        mock_vector_store = Mock(spec=VectorStoreManager)
        mock_retriever = Mock()
        mock_docs = [Document(page_content="Apple content", metadata={"title": "Apple", "ticker": "AAPL", "link": "http://test.com"})]
        mock_retriever.invoke = Mock(return_value=mock_docs)
        mock_retriever.get_relevant_documents = Mock(return_value=mock_docs)
        mock_vector_store.get_retriever.return_value = mock_retriever
        
        with patch.object(VectorStoreManager, 'initialize_vector_store'):
            rag_chain = RAGChain(mock_vector_store)
            
            with patch('core.rag_chain._invoke_llm_with_backoff') as mock_invoke:
                mock_invoke.return_value = {
                    "answer": "Test answer",
                    "context": mock_docs,
                    "citations": []
                }
                
                result = rag_chain.query("What did Apple announce?", k=3)
                
                self.assertIn("answer", result)
                self.assertIn("sources", result)
                self.assertIn("citations", result)
    
    @patch('core.rag_chain.initialize_llm_with_fallback')
    def test_rag_chain_with_tracker(self, mock_init_llm):
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value="Answer")
        mock_structured_llm = Mock()
        mock_structured_llm.invoke = Mock(return_value=Mock(citations=[]))
        mock_llm.with_structured_output = Mock(return_value=mock_structured_llm)
        mock_init_llm.return_value = mock_llm
        
        mock_vector_store = Mock(spec=VectorStoreManager)
        mock_retriever = Mock()
        mock_docs = [Document(page_content="Content", metadata={"title": "Test", "ticker": "AAPL"})]
        mock_retriever.invoke = Mock(return_value=mock_docs)
        mock_retriever.get_relevant_documents = Mock(return_value=mock_docs)
        mock_vector_store.get_retriever.return_value = mock_retriever
        
        rag_chain = RAGChain(mock_vector_store)
        tracker = ProcessingStepsTracker()
        
        with patch('core.rag_chain._invoke_llm_with_backoff') as mock_invoke:
            mock_invoke.return_value = {
                "answer": "Answer",
                "context": mock_docs,
                "citations": []
            }
            
            result = rag_chain.query("Test question", tracker=tracker)
            
            self.assertIn("answer", result)
            self.assertIn("sources", result)
            self.assertIn("citations", result)
            
            mock_vector_store.get_retriever.assert_called_once()
            call_kwargs = mock_vector_store.get_retriever.call_args[1]
            self.assertEqual(call_kwargs.get('tracker'), tracker)
    
    @unittest.skip("Requires OpenAI API key and vector store setup")
    def test_full_rag_pipeline_with_query(self):
        documents = load_stock_news(self.temp_file.name)
        
        vector_store = VectorStoreManager(use_contextual_retrieval=False)
        vector_store.initialize_vector_store(documents, force_recreate=True)
        
        rag_chain = RAGChain(vector_store)
        
        result = rag_chain.query("What did Apple announce?")
        
        self.assertIn("answer", result)
        self.assertIn("sources", result)
        self.assertIn("citations", result)
        self.assertIsInstance(result["answer"], str)
        self.assertIsInstance(result["sources"], list)
        self.assertIsInstance(result["citations"], list)
        self.assertGreater(len(result["answer"]), 0)


if __name__ == '__main__':
    unittest.main()
