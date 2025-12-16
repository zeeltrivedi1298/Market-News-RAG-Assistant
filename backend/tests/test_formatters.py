import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from utils.formatters import format_docs_with_metadata, format_rag_result, _extract_citations
from models.citation_models import QuotedCitations, Citation


class TestFormatDocsWithMetadata(unittest.TestCase):
    def test_format_empty_list(self):
        result = format_docs_with_metadata([])
        self.assertEqual(result, "")
    
    def test_format_single_document(self):
        doc = Document(
            page_content="Test content",
            metadata={
                "id": "test-id",
                "title": "Test Title",
                "ticker": "AAPL",
                "source": "test_source",
                "chunk_index": 1,
                "total_chunks": 5
            }
        )
        
        result = format_docs_with_metadata([doc])
        
        self.assertIn("Test content", result)
        self.assertIn("test-id", result)
        self.assertIn("Test Title", result)
        self.assertIn("AAPL", result)
        self.assertIn("test_source", result)
        self.assertIn("1/5", result)
    
    def test_format_multiple_documents(self):
        docs = [
            Document(page_content="Content 1", metadata={"id": "1", "title": "Title 1"}),
            Document(page_content="Content 2", metadata={"id": "2", "title": "Title 2"})
        ]
        
        result = format_docs_with_metadata(docs)
        
        self.assertIn("Content 1", result)
        self.assertIn("Content 2", result)
        self.assertIn("Title 1", result)
        self.assertIn("Title 2", result)
    
    def test_skips_empty_content(self):
        docs = [
            Document(page_content="", metadata={"id": "1"}),
            Document(page_content="Valid content", metadata={"id": "2"})
        ]
        
        result = format_docs_with_metadata(docs)
        
        self.assertNotIn("id: 1", result.lower())
        self.assertIn("Valid content", result)
        self.assertIn("id: 2", result.lower())
    
    def test_handles_missing_metadata(self):
        doc = Document(page_content="Content", metadata={})
        
        result = format_docs_with_metadata([doc])
        
        self.assertIn("Content", result)
        self.assertIn("Unknown", result)
    
    def test_handles_invalid_document(self):
        class InvalidDoc:
            pass
        
        result = format_docs_with_metadata([InvalidDoc()])
        self.assertEqual(result, "")


class TestFormatRagResult(unittest.TestCase):
    def test_format_with_all_fields(self):
        doc = Document(
            page_content="Content",
            metadata={"title": "Article Title", "ticker": "AAPL", "link": "http://example.com"}
        )
        
        result = format_rag_result({
            "answer": "Test answer",
            "context": [doc],
            "citations": []
        })
        
        self.assertEqual(result["answer"], "Test answer")
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["title"], "Article Title")
        self.assertEqual(result["sources"][0]["ticker"], "AAPL")
        self.assertEqual(result["sources"][0]["link"], "http://example.com")
        self.assertEqual(len(result["citations"]), 0)
    
    def test_format_with_empty_answer(self):
        result = format_rag_result({
            "answer": "",
            "context": [],
            "citations": []
        })
        
        self.assertIn("couldn't generate", result["answer"].lower())
    
    def test_format_removes_duplicate_sources(self):
        doc1 = Document(
            page_content="Content 1",
            metadata={"title": "Same Title", "ticker": "AAPL", "link": "http://example.com"}
        )
        doc2 = Document(
            page_content="Content 2",
            metadata={"title": "Same Title", "ticker": "AAPL", "link": "http://example.com"}
        )
        
        result = format_rag_result({
            "answer": "Answer",
            "context": [doc1, doc2],
            "citations": []
        })
        
        self.assertEqual(len(result["sources"]), 1)
    
    def test_format_handles_missing_metadata(self):
        doc = Document(page_content="Content", metadata={})
        
        result = format_rag_result({
            "answer": "Answer",
            "context": [doc],
            "citations": []
        })
        
        self.assertEqual(result["sources"][0]["title"], "Unknown")
        self.assertEqual(result["sources"][0]["ticker"], "Unknown")
        self.assertEqual(result["sources"][0]["link"], "")


class TestExtractCitations(unittest.TestCase):
    def test_extract_from_quoted_citations_object(self):
        citation1 = Citation(
            id="id1",
            source="source1",
            title="Title 1",
            ticker="AAPL",
            quotes="Quote 1"
        )
        citation2 = Citation(
            id="id2",
            source="source2",
            title="Title 2",
            ticker="MSFT",
            quotes="Quote 2"
        )
        
        quoted_citations = QuotedCitations(citations=[citation1, citation2])
        result = _extract_citations(quoted_citations)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], "id1")
        self.assertEqual(result[0]["title"], "Title 1")
        self.assertEqual(result[1]["id"], "id2")
        self.assertEqual(result[1]["title"], "Title 2")
    
    def test_extract_from_list(self):
        citations_list = [
            {
                "id": "id1",
                "source": "source1",
                "title": "Title 1",
                "ticker": "AAPL",
                "page": 1,
                "quotes": "Quote 1"
            },
            {
                "id": "id2",
                "source": "source2",
                "title": "Title 2",
                "ticker": "MSFT",
                "page": 2,
                "quotes": "Quote 2"
            }
        ]
        
        result = _extract_citations(citations_list)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], "id1")
        self.assertEqual(result[1]["id"], "id2")
    
    def test_extract_from_none(self):
        result = _extract_citations(None)
        self.assertEqual(result, [])
    
    def test_extract_from_empty_list(self):
        result = _extract_citations([])
        self.assertEqual(result, [])
    
    def test_extract_handles_invalid_citation(self):
        class InvalidCitation:
            pass
        
        result = _extract_citations([InvalidCitation()])
        self.assertEqual(result, [])
    
    def test_extract_handles_missing_fields(self):
        citation_dict = {
            "id": "id1",
            "title": "Title"
        }
        
        result = _extract_citations([citation_dict])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "id1")
        self.assertEqual(result[0]["source"], "")
        self.assertEqual(result[0]["ticker"], "")
        self.assertEqual(result[0]["page"], 1)
        self.assertEqual(result[0]["quotes"], "")


if __name__ == '__main__':
    unittest.main()

