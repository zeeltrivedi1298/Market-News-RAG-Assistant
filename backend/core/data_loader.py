import json
import uuid
import os
from typing import List
from langchain_core.documents import Document

def _load_json(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file: {str(e)}")

def load_stock_news(file_path: str) -> List[Document]:
    data = _load_json(file_path)
    if not isinstance(data, dict) or not data:
        raise ValueError("Data file must contain a non-empty dictionary")
    
    documents = []
    for ticker, articles in data.items():
        if not ticker or not isinstance(articles, list):
            continue
        for article in articles:
            if not isinstance(article, dict):
                continue
            title = article.get("title", "").strip()
            full_text = article.get("full_text", "").strip()
            if not title and not full_text:
                continue
            
            documents.append(Document(
                page_content=f"Title: {title}\n\n{full_text}" if title else full_text,
                metadata={
                    "id": str(uuid.uuid4()),
                    "ticker": ticker,
                    "title": title,
                    "link": article.get("link", "").strip(),
                    "source": "stock_news",
                    "page": 1
                }
            ))
    
    if not documents:
        raise ValueError("No valid documents found in data file")
    return documents

def get_tickers(file_path: str) -> List[str]:
    data = _load_json(file_path)
    if not isinstance(data, dict):
        raise ValueError("Data file must contain a dictionary")
    tickers = [ticker for ticker in data.keys() if ticker and ticker.strip()]
    if not tickers:
        raise ValueError("No valid tickers found in data file")
    return tickers
