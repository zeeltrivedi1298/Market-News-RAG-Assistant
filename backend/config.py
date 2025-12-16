import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

try:
    load_dotenv()
except Exception as e:
    logger.warning(f"Failed to load .env file: {str(e)}")

class Config:
    _api_key = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_KEY = _api_key.strip() if isinstance(_api_key, str) else ""
    
    _llm_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    LLM_MODEL = _llm_model.strip() if isinstance(_llm_model, str) else "gpt-3.5-turbo"
    
    _embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_MODEL = _embedding_model.strip() if isinstance(_embedding_model, str) else "all-MiniLM-L6-v2"
    
     _llm_fallback_model = os.getenv("LLM_FALLBACK_MODEL", "gpt-4")
    LLM_FALLBACK_MODEL = _llm_fallback_model.strip() if isinstance(_llm_fallback_model, str) and _llm_fallback_model.strip() else "gpt-4"
    
    _embedding_fallback_model = os.getenv("EMBEDDING_FALLBACK_MODEL", "all-MiniLM-L12-v2")
    EMBEDDING_FALLBACK_MODEL = _embedding_fallback_model.strip() if isinstance(_embedding_fallback_model, str) else "all-MiniLM-L12-v2"
    
    ENABLE_LLM_FALLBACK = os.getenv("ENABLE_LLM_FALLBACK", "true").lower() == "true"
    ENABLE_EMBEDDING_FALLBACK = os.getenv("ENABLE_EMBEDDING_FALLBACK", "true").lower() == "true"
    
    _max_retries = os.getenv("MAX_RETRIES", "3")
    try:
        MAX_RETRIES = int(_max_retries) if isinstance(_max_retries, str) and _max_retries.strip().isdigit() else 3
        MAX_RETRIES = max(0, min(10, MAX_RETRIES))
    except (ValueError, TypeError):
        MAX_RETRIES = 3
    
    _initial_retry_delay = os.getenv("INITIAL_RETRY_DELAY", "1.0")
    try:
        INITIAL_RETRY_DELAY = float(_initial_retry_delay) if isinstance(_initial_retry_delay, str) else 1.0
        INITIAL_RETRY_DELAY = max(0.1, min(10.0, INITIAL_RETRY_DELAY))
    except (ValueError, TypeError):
        INITIAL_RETRY_DELAY = 1.0
    
    _backoff_factor = os.getenv("BACKOFF_FACTOR", "2.0")
    try:
        BACKOFF_FACTOR = float(_backoff_factor) if isinstance(_backoff_factor, str) else 2.0
        BACKOFF_FACTOR = max(1.0, min(10.0, BACKOFF_FACTOR))
    except (ValueError, TypeError):
        BACKOFF_FACTOR = 2.0
    
    _max_retry_delay = os.getenv("MAX_RETRY_DELAY", "60.0")
    try:
        MAX_RETRY_DELAY = float(_max_retry_delay) if isinstance(_max_retry_delay, str) else 60.0
        MAX_RETRY_DELAY = max(1.0, min(300.0, MAX_RETRY_DELAY))
    except (ValueError, TypeError):
        MAX_RETRY_DELAY = 60.0
    
    _chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_PERSIST_DIR = _chroma_dir.strip() if isinstance(_chroma_dir, str) else "./chroma_db"
    
    _host = os.getenv("HOST", "0.0.0.0")
    HOST = _host.strip() if isinstance(_host, str) else "0.0.0.0"
    
    _port = os.getenv("PORT", "8000")
    try:
        PORT = int(_port) if isinstance(_port, str) and _port.strip().isdigit() else 8000
        if PORT < 1 or PORT > 65535:
            PORT = 8000
    except (ValueError, TypeError):
        PORT = 8000
    
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    
    _chunk_size_tokens = os.getenv("CHUNK_SIZE_TOKENS", "400")
    try:
        CHUNK_SIZE_TOKENS = int(_chunk_size_tokens) if isinstance(_chunk_size_tokens, str) and _chunk_size_tokens.strip().isdigit() else 400
        if CHUNK_SIZE_TOKENS < 100 or CHUNK_SIZE_TOKENS > 1000:
            CHUNK_SIZE_TOKENS = 400
    except (ValueError, TypeError):
        CHUNK_SIZE_TOKENS = 400
    
    _chunk_overlap_tokens = os.getenv("CHUNK_OVERLAP_TOKENS", "60")
    try:
        CHUNK_OVERLAP_TOKENS = int(_chunk_overlap_tokens) if isinstance(_chunk_overlap_tokens, str) and _chunk_overlap_tokens.strip().isdigit() else 60
        if CHUNK_OVERLAP_TOKENS < 0 or CHUNK_OVERLAP_TOKENS > CHUNK_SIZE_TOKENS:
            CHUNK_OVERLAP_TOKENS = 60
    except (ValueError, TypeError):
        CHUNK_OVERLAP_TOKENS = 60
    
    USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "false").lower() == "true"
    _hybrid_alpha = os.getenv("HYBRID_SEARCH_ALPHA", "0.5")
    try:
        HYBRID_SEARCH_ALPHA = float(_hybrid_alpha)
        HYBRID_SEARCH_ALPHA = max(0.0, min(1.0, HYBRID_SEARCH_ALPHA))
    except (ValueError, TypeError):
        HYBRID_SEARCH_ALPHA = 0.5
    RERANK_TOP_K_MULTIPLIER = int(os.getenv("RERANK_TOP_K_MULTIPLIER", "2"))

