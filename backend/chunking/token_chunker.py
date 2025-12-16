import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config

logger = logging.getLogger(__name__)

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available. Install with: pip install tiktoken. Falling back to character-based chunking.")


class TokenChunker:
    def __init__(self):
        if TIKTOKEN_AVAILABLE:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                length_function = lambda text: len(encoding.encode(text))
                chunk_size = Config.CHUNK_SIZE_TOKENS
                chunk_overlap = Config.CHUNK_OVERLAP_TOKENS
            except Exception as e:
                logger.warning(f"Failed to initialize tiktoken: {e}. Using character-based chunking.")
                length_function = len
                chunk_size = Config.CHUNK_SIZE_TOKENS * 4
                chunk_overlap = Config.CHUNK_OVERLAP_TOKENS * 4
        else:
            length_function = len
            chunk_size = Config.CHUNK_SIZE_TOKENS * 4
            chunk_overlap = Config.CHUNK_OVERLAP_TOKENS * 4
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            keep_separator=True,
        )
    
    def chunk(self, documents: List[Document]) -> List[Document]:
        return self.text_splitter.split_documents(documents)

