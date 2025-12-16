import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.citation_models import Citation, QuotedCitations


class TestCitationModels(unittest.TestCase):
    def test_citation_creation(self):
        citation = Citation(
            id="test-id-123",
            source="stock_news",
            title="Test Article",
            ticker="AAPL",
            quotes="This is a test quote from the article."
        )
        
        self.assertEqual(citation.id, "test-id-123")
        self.assertEqual(citation.source, "stock_news")
        self.assertEqual(citation.title, "Test Article")
        self.assertEqual(citation.ticker, "AAPL")
        self.assertEqual(citation.quotes, "This is a test quote from the article.")
        self.assertEqual(citation.page, 1)
    
    def test_citation_with_page(self):
        citation = Citation(
            id="test-id",
            source="source",
            title="Title",
            ticker="MSFT",
            page=2,
            quotes="Quote text"
        )
        
        self.assertEqual(citation.page, 2)
    
    def test_quoted_citations_creation(self):
        citation1 = Citation(
            id="id1",
            source="source1",
            title="Title1",
            ticker="AAPL",
            quotes="Quote1"
        )
        
        citation2 = Citation(
            id="id2",
            source="source2",
            title="Title2",
            ticker="MSFT",
            quotes="Quote2"
        )
        
        quoted_citations = QuotedCitations(citations=[citation1, citation2])
        
        self.assertEqual(len(quoted_citations.citations), 2)
        self.assertEqual(quoted_citations.citations[0].id, "id1")
        self.assertEqual(quoted_citations.citations[1].id, "id2")
    
    def test_citation_quotes_required(self):
        citation = Citation(
            id="test",
            source="source",
            title="title",
            ticker="AAPL",
            quotes="Valid quote"
        )
        
        self.assertIsNotNone(citation.quotes)
        self.assertGreater(len(citation.quotes), 0)


if __name__ == '__main__':
    unittest.main()

