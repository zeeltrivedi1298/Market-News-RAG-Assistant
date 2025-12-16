import logging
from typing import List
from langchain_core.documents import Document
from chunking.token_chunker import TokenChunker
from chunking.structure_chunker import StructureChunker
from chunking.semantic_chunker import SemanticChunker

logger = logging.getLogger(__name__)


class ChunkingPipeline:
    def __init__(self, embeddings=None):
        if not embeddings:
            raise ValueError("embeddings are required for semantic chunking.")
        
        self.token_chunker = TokenChunker()
        self.structure_chunker = StructureChunker()
        
        try:
            self.semantic_chunker = SemanticChunker(embeddings, similarity_threshold=0.5)
            logger.info("Semantic chunker initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize semantic chunker: {e}")
            raise RuntimeError(f"Failed to initialize semantic chunker: {e}") from e
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        try:
            logger.info("Applying semantic chunking...")
            documents = self.semantic_chunker.chunk_by_semantics(documents)
            logger.info(f"Semantic chunking created {len(documents)} semantic chunks")
        except Exception as e:
            logger.error(f"Semantic chunking failed: {e}")
            raise RuntimeError(f"Semantic chunking failed: {e}") from e
        
        logger.info("Splitting documents into chunks using token-based, structure-aware chunking...")
        all_chunks = []
        
        for doc in documents:
            doc_chunks = self.token_chunker.chunk([doc])
            enhanced_chunks = self.structure_chunker.enhance_chunk_metadata(
                doc_chunks,
                doc.metadata.get('title', 'Unknown'),
                doc.metadata.get('id', 'unknown')
            )
            all_chunks.extend(enhanced_chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks


def create_chunker(embeddings=None) -> ChunkingPipeline:
    return ChunkingPipeline(embeddings=embeddings)

