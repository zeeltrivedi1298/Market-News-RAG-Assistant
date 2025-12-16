import logging
from typing import List, Dict, Any
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def format_docs_with_metadata(docs: List[Document]) -> str:
    if not docs:
        return ""
    
    formatted_docs = []
    for doc in docs:
        if not hasattr(doc, 'page_content') or not hasattr(doc, 'metadata'):
            continue
        
        metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
        page_content = doc.page_content or ""
        
        if not page_content.strip():
            continue
        
        section_title = metadata.get('section_title', metadata.get('title', 'Unknown'))
        parent_title = metadata.get('parent_title', metadata.get('title', 'Unknown'))
        chunk_index = metadata.get('chunk_index', '?')
        total_chunks = metadata.get('total_chunks', '?')
        
        formatted_doc = f"""Context Article ID: {metadata.get('article_id', metadata.get('id', 'unknown'))}
Context Article Source: {metadata.get('source', 'unknown')}
Context Article Title: {parent_title}
Context Section: {section_title}
Context Article Ticker: {metadata.get('ticker', 'Unknown')}
Context Chunk: {chunk_index}/{total_chunks}
Context Article Page: {metadata.get('page', 1)}
Context Article Details: {page_content}
"""
        formatted_docs.append(formatted_doc)
    
    return "\n\n".join(formatted_docs) if formatted_docs else ""

def format_rag_result(result: Dict[str, Any]) -> Dict[str, Any]:
    sources = []
    seen_sources = set()
    
    for doc in result.get("context", []):
        if not hasattr(doc, 'metadata'):
            continue
        metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
        title = metadata.get("title", "Unknown")
        link = metadata.get("link", "")
        ticker = metadata.get("ticker", "Unknown")
        
        source_key = (title.lower().strip(), link.lower().strip() if link else "")
        
        if source_key in seen_sources:
            continue
        
        seen_sources.add(source_key)
        sources.append({
            "title": title,
            "ticker": ticker,
            "link": link,
        })
    
    citations = _extract_citations(result.get("citations"))
    
    answer = result.get("answer", "")
    if not answer:
        answer = "I'm sorry, I couldn't generate an answer. Please try rephrasing your question."
    
    answer = _remove_star_ratings(answer)
    
    return {
        "answer": answer,
        "sources": sources,
        "citations": citations
    }


def _remove_star_ratings(text: str) -> str:
    import re
    if not text:
        return text
    
    text = re.sub(r'[★☆]+', '', text)
    text = re.sub(r'\*{2,}', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def _extract_citations(citations_obj) -> List[Dict[str, Any]]:
    if not citations_obj:
        return []
    
    if hasattr(citations_obj, 'citations'):
        citations_list = citations_obj.citations
    elif isinstance(citations_obj, list):
        citations_list = citations_obj
    else:
        return []
    
    formatted_citations = []
    for citation in citations_list:
        try:
            if isinstance(citation, dict):
                formatted_citations.append({
                    "id": str(citation.get("id", "")),
                    "source": str(citation.get("source", "")),
                    "title": str(citation.get("title", "Unknown")),
                    "ticker": str(citation.get("ticker", "")),
                    "page": int(citation.get("page", 1)),
                    "quotes": str(citation.get("quotes", ""))
                })
            elif hasattr(citation, 'id'):
                formatted_citations.append({
                    "id": str(citation.id),
                    "source": str(citation.source),
                    "title": str(citation.title),
                    "ticker": str(citation.ticker),
                    "page": int(getattr(citation, 'page', 1)),
                    "quotes": str(citation.quotes)
                })
        except Exception as e:
            logger.warning(f"Failed to format citation: {e}")
            continue
    
    return formatted_citations
