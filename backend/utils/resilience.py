import logging
import time
from typing import Optional, Callable, Any
from functools import wraps
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import Config

logger = logging.getLogger(__name__)


def _is_retryable_error(error: Exception) -> bool:
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()
    
    retryable_indicators = [
        '429', 'rate limit', 'ratelimit', 'too many requests',
        '500', '502', '503', '504', 'server error', 'internal error',
        'timeout', 'connection', 'network', 'temporary', 'unavailable'
    ]
    
    return any(indicator in error_str or indicator in error_type for indicator in retryable_indicators)


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0
):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if not _is_retryable_error(e):
                        logger.error(f"Non-retryable error in {func.__name__}: {e}")
                        raise
                    
                    if attempt < max_retries:
                        delay = min(initial_delay * (backoff_factor ** attempt), max_delay)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}: {e}")
            
            raise last_exception
        return wrapper
    return decorator


def initialize_llm_with_fallback(
    primary_model: str = None,
    fallback_model: str = None,
    api_key: str = None,
    temperature: float = 0,
    **kwargs
) -> Optional[ChatOpenAI]:
    primary_model = primary_model or Config.LLM_MODEL
    fallback_model = fallback_model or Config.LLM_FALLBACK_MODEL
    api_key = api_key or Config.OPENAI_API_KEY
    
    if not api_key:
        logger.warning("No OpenAI API key provided for LLM initialization")
        return None
    
    if not Config.ENABLE_LLM_FALLBACK:
        try:
            return ChatOpenAI(
                model=primary_model,
                temperature=temperature,
                api_key=api_key,
                max_retries=Config.MAX_RETRIES,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to initialize primary LLM: {e}")
            return None
    
    @retry_with_backoff(
        max_retries=Config.MAX_RETRIES,
        initial_delay=Config.INITIAL_RETRY_DELAY,
        backoff_factor=Config.BACKOFF_FACTOR,
        max_delay=Config.MAX_RETRY_DELAY
    )
    def _try_llm(model_name: str) -> ChatOpenAI:
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            max_retries=0,
            **kwargs
        )
    
    try:
        logger.info(f"Initializing primary LLM: {primary_model}")
        return _try_llm(primary_model)
    except Exception as e:
        if primary_model == fallback_model:
            logger.error(f"LLM initialization failed and no different fallback model configured: {e}")
            return None
        logger.warning(f"Primary LLM ({primary_model}) failed: {e}, trying fallback...")
        try:
            logger.info(f"Initializing fallback LLM: {fallback_model}")
            llm = _try_llm(fallback_model)
            logger.info(f"Successfully initialized fallback LLM: {fallback_model}")
            return llm
        except Exception as e2:
            logger.error(f"Both primary and fallback LLMs failed. Primary: {e}, Fallback: {e2}")
            return None


def initialize_embeddings_with_fallback(
    primary_model: str = None,
    fallback_model: str = None,
    **kwargs
) -> HuggingFaceEmbeddings:
    primary_model = primary_model or Config.EMBEDDING_MODEL
    fallback_model = fallback_model or Config.EMBEDDING_FALLBACK_MODEL
    
    if not Config.ENABLE_EMBEDDING_FALLBACK:
        try:
            return HuggingFaceEmbeddings(model_name=primary_model, **kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize primary embeddings: {e}")
            raise RuntimeError(f"Embeddings initialization failed: {e}")
    
    @retry_with_backoff(
        max_retries=Config.MAX_RETRIES,
        initial_delay=Config.INITIAL_RETRY_DELAY,
        backoff_factor=Config.BACKOFF_FACTOR,
        max_delay=Config.MAX_RETRY_DELAY
    )
    def _try_embeddings(model_name: str) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(model_name=model_name, **kwargs)
    
    try:
        logger.info(f"Initializing primary embeddings: {primary_model}")
        return _try_embeddings(primary_model)
    except Exception as e:
        if primary_model == fallback_model:
            logger.error(f"Embeddings initialization failed and no different fallback model configured: {e}")
            raise RuntimeError(f"Embeddings initialization failed: {e}")
        logger.warning(f"Primary embeddings ({primary_model}) failed: {e}, trying fallback...")
        try:
            logger.info(f"Initializing fallback embeddings: {fallback_model}")
            embeddings = _try_embeddings(fallback_model)
            logger.info(f"Successfully initialized fallback embeddings: {fallback_model}")
            return embeddings
        except Exception as e2:
            logger.error(f"Both primary and fallback embeddings failed. Primary: {e}, Fallback: {e2}")
            raise RuntimeError("Both primary and fallback embeddings failed. Cannot proceed.")


