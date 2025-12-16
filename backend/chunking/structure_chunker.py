import re
from typing import List
from langchain_core.documents import Document


class StructureChunker:
    @staticmethod
    def extract_section_title(chunk_text: str, fallback_title: str) -> str:
        if not chunk_text:
            return fallback_title
        
        heading_match = re.search(r'^#+\s+(.+)$', chunk_text, re.MULTILINE)
        if heading_match:
            return heading_match.group(1).strip()
        
        for line in chunk_text.split('\n')[:3]:
            line = line.strip()
            if 10 < len(line) < 100 and (line.isupper() or (line.istitle() and not line.endswith('.'))):
                return line
        
        first_sentence = chunk_text.split('.')[0].strip()
        if 10 < len(first_sentence) < 100 and not first_sentence.endswith('.'):
            return first_sentence
        
        return fallback_title
    
    @staticmethod
    def enhance_chunk_metadata(chunks: List[Document], doc_title: str, doc_id: str) -> List[Document]:
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            section_title = StructureChunker.extract_section_title(chunk.page_content, doc_title)
            enhanced_metadata = (chunk.metadata.copy() if chunk.metadata else {})
            enhanced_metadata.update({
                'chunk_index': i,
                'total_chunks': len(chunks),
                'section_title': section_title,
                'parent_title': doc_title,
                'article_id': doc_id,
            })
            enhanced_chunks.append(Document(
                page_content=chunk.page_content,
                metadata=enhanced_metadata
            ))
        return enhanced_chunks

