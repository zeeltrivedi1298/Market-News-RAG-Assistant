import unittest
import json
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_loader import load_stock_news, get_tickers


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.sample_data = {
            "AAPL": [
                {
                    "title": "Apple Announces New iPhone",
                    "link": "https://example.com/apple",
                    "full_text": "Apple Inc. announced a new iPhone model with advanced features."
                }
            ],
            "MSFT": [
                {
                    "title": "Microsoft Cloud Growth",
                    "link": "https://example.com/microsoft",
                    "full_text": "Microsoft reported strong cloud revenue growth this quarter."
                }
            ]
        }
    
    def test_loads_valid_stock_news(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_data, f)
            temp_path = f.name
        
        try:
            documents = load_stock_news(temp_path)
            
            self.assertEqual(len(documents), 2)
            self.assertEqual(documents[0].metadata["ticker"], "AAPL")
            self.assertEqual(documents[0].metadata["title"], "Apple Announces New iPhone")
            self.assertIn("id", documents[0].metadata)
            self.assertIn("Apple Announces New iPhone", documents[0].page_content)
        finally:
            os.unlink(temp_path)
    
    def test_extracts_tickers_correctly(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_data, f)
            temp_path = f.name
        
        try:
            tickers = get_tickers(temp_path)
            self.assertEqual(set(tickers), {"AAPL", "MSFT"})
        finally:
            os.unlink(temp_path)
    
    def test_handles_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_stock_news("nonexistent_file.json")
    
    def test_handles_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name
        
        try:
            with self.assertRaises(ValueError):
                load_stock_news(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_handles_empty_data_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({}, f)
            temp_path = f.name
        
        try:
            with self.assertRaises(ValueError):
                load_stock_news(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_skips_articles_without_content(self):
        data_with_empty = {
            "AAPL": [
                {"title": "", "full_text": ""},
                {"title": "Valid Article", "full_text": "This has content"}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data_with_empty, f)
            temp_path = f.name
        
        try:
            documents = load_stock_news(temp_path)
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].metadata["title"], "Valid Article")
        finally:
            os.unlink(temp_path)
    
    def test_handles_articles_with_partial_content(self):
        test_cases = [
            {
                "name": "title_only",
                "data": {"AAPL": [{"title": "Title Only Article", "full_text": ""}]},
                "expected_in_content": "Title Only Article"
            },
            {
                "name": "text_only",
                "data": {"AAPL": [{"title": "", "full_text": "This is the article content"}]},
                "expected_in_content": "This is the article content"
            }
        ]
        
        for case in test_cases:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(case["data"], f)
                temp_path = f.name
            
            try:
                documents = load_stock_news(temp_path)
                self.assertEqual(len(documents), 1, f"Failed for case: {case['name']}")
                self.assertIn(case["expected_in_content"], documents[0].page_content, f"Failed for case: {case['name']}")
            finally:
                os.unlink(temp_path)
    
    def test_assigns_unique_ids_to_documents(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_data, f)
            temp_path = f.name
        
        try:
            documents = load_stock_news(temp_path)
            ids = [doc.metadata["id"] for doc in documents]
            self.assertEqual(len(ids), len(set(ids)), "All IDs should be unique")
        finally:
            os.unlink(temp_path)
    
    def test_handles_missing_link_field(self):
        data_no_link = {
            "AAPL": [
                {"title": "Article", "full_text": "Content"}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data_no_link, f)
            temp_path = f.name
        
        try:
            documents = load_stock_news(temp_path)
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].metadata["link"], "")
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()
