import os
import logging
from collections import defaultdict
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import Config
from chunking.factory import create_chunker
from utils.resilience import initialize_llm_with_fallback, initialize_embeddings_with_fallback

logger = logging.getLogger(__name__)


class VectorStoreManager:
    def __init__(self, use_contextual_retrieval: bool = True):
        self.embeddings = initialize_embeddings_with_fallback(
            model_kwargs={'device': 'cpu'}
        )
        
        self.chunker = create_chunker(embeddings=self.embeddings)
        self.vector_store = None
        self.use_contextual_retrieval = use_contextual_retrieval
        self.llm = None
        self.chunks = []
        if use_contextual_retrieval:
            self._initialize_llm()
    
    def _initialize_llm(self):
        self.llm = initialize_llm_with_fallback(temperature=0)
        if not self.llm:
            logger.warning("Both primary and fallback LLMs failed. Contextual retrieval disabled.")
            self.use_contextual_retrieval = False
    
    def generate_chunk_context(self, document: str, chunk: str) -> str:
        if not self.llm or not document or not chunk:
            return ""
        
        MAX_DOC_LENGTH = 50000
        MAX_CHUNK_LENGTH = 5000
        
        if len(document) > MAX_DOC_LENGTH:
            document = document[:MAX_DOC_LENGTH] + "..."
        if len(chunk) > MAX_CHUNK_LENGTH:
            chunk = chunk[:MAX_CHUNK_LENGTH] + "..."
        
        prompt = self._get_context_prompt()
        
        try:
            prompt_template = ChatPromptTemplate.from_template(prompt)
            chain = prompt_template | self.llm | StrOutputParser()
            return chain.invoke({'article': document, 'chunk': chunk})
        except Exception as e:
            logger.error(f"Error generating chunk context: {e}")
            return ""
    
    def _get_context_prompt(self) -> str:
        return """You are an AI assistant specializing in financial news analysis.
Your task is to provide brief, relevant context for a chunk of text based on the following financial news article.

Here is the financial news article:
<article>
{article}
</article>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Provide a concise context (3-4 sentences max) for this chunk, considering the following guidelines:

- Give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
- Answer only with the succinct context and nothing else.
- Context should be mentioned like 'Focuses on ....' do not mention 'this chunk or section focuses on...'

Context:"""
    
    def initialize_vector_store(self, documents: List[Document], force_recreate: bool = False):
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        valid_docs = [doc for doc in documents if isinstance(doc, Document) and doc.page_content]
        if not valid_docs:
            raise ValueError("No valid documents found")
        
        persist_directory = Config.CHROMA_PERSIST_DIR
        
        if not force_recreate and os.path.exists(persist_directory):
            try:
                self.vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info(f"Loaded existing vector store from {persist_directory}")
                if Config.USE_HYBRID_SEARCH:
                    try:
                        all_docs = self.vector_store.get(limit=None)
                        if all_docs and 'ids' in all_docs and len(all_docs['ids']) > 0:
                            self.chunks = []
                            for i, doc_id in enumerate(all_docs['ids']):
                                metadata = all_docs.get('metadatas', [{}])[i] if 'metadatas' in all_docs and i < len(all_docs.get('metadatas', [])) else {}
                                page_content = all_docs.get('documents', [''])[i] if 'documents' in all_docs and i < len(all_docs.get('documents', [])) else ''
                                if page_content:
                                    self.chunks.append(Document(
                                        page_content=page_content,
                                        metadata=metadata
                                    ))
                            logger.info(f"Loaded {len(self.chunks)} chunks for hybrid search")
                        else:
                            logger.warning("No documents found in vector store for hybrid search")
                            self.chunks = []
                    except Exception as e:
                        logger.warning(f"Failed to load chunks for hybrid search: {e}")
                        self.chunks = []
                return
            except Exception as e:
                logger.warning(f"Failed to load existing vector store: {str(e)}. Recreating...")
        
        chunks = self.chunker.chunk_documents(valid_docs)
        
        if not chunks:
            raise ValueError("No chunks created from documents")
        
        if self.use_contextual_retrieval and self.llm:
            chunks = self._add_contextual_summaries(chunks)
        
        self.chunks = chunks
        
        logger.info("Creating vector store...")
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        logger.info(f"Vector store created and persisted to {persist_directory}")
    
    def get_retriever(self, k: int = 3, ticker_filter: str = None, tracker=None):
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call initialize_vector_store() first.")
        
        k = max(1, min(20, k if isinstance(k, int) else 3))
        search_kwargs = {"k": k}
        if ticker_filter:
            search_kwargs["filter"] = {"ticker": ticker_filter.strip().upper()}
        
        if tracker:
            step_num = tracker.start_step(
                "Building Retriever",
                "Setting up document retriever with search parameters",
                {"k": k, "ticker_filter": ticker_filter or "None"}
            )
        
        base_retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
        
        hybrid_search_enabled = Config.USE_HYBRID_SEARCH and self.chunks
        
        if hybrid_search_enabled:
            try:
                from retrieval.hybrid_retriever import HybridRetriever
                hybrid_retriever = HybridRetriever(
                    vector_retriever=base_retriever,
                    documents=self.chunks,
                    alpha=Config.HYBRID_SEARCH_ALPHA
                )
                logger.info("Using hybrid search (vector + BM25)")
                if tracker:
                    tracker.complete_step(step_num, {
                        "retriever_type": "Hybrid (Vector + BM25)",
                        "use_hybrid_search": True,
                        "alpha": Config.HYBRID_SEARCH_ALPHA,
                        "total_chunks": len(self.chunks)
                    })
                return hybrid_retriever
            except Exception as e:
                logger.warning(f"Failed to create hybrid retriever, falling back to vector-only: {e}")
                if tracker:
                    tracker.complete_step(step_num, {
                        "retriever_type": "Vector-only (fallback)",
                        "use_hybrid_search": False,
                        "error": str(e),
                        "reason": "Hybrid search failed to initialize"
                    })
                return base_retriever
        
        if tracker:
            reason = "Not enabled in config" if not Config.USE_HYBRID_SEARCH else "No chunks available"
            tracker.complete_step(step_num, {
                "retriever_type": "Vector-only",
                "use_hybrid_search": False,
                "reason": reason
            })
        
        return base_retriever
    
    def _add_contextual_summaries(self, chunks: List[Document]) -> List[Document]:
        logger.info("Generating contextual summaries for chunks...")
        
        chunks_by_doc = self._group_chunks_by_document(chunks)
        contextual_chunks = []
        total_chunks = len(chunks)
        processed = 0
        
        for doc_id, doc_chunks in chunks_by_doc.items():
            if not doc_chunks:
                continue
            
            original_doc = self._reconstruct_document(doc_chunks)
            if not original_doc.strip():
                contextual_chunks.extend(doc_chunks)
                processed += len(doc_chunks)
                continue
            
            for chunk in doc_chunks:
                context = self.generate_chunk_context(original_doc, chunk.page_content)
                contextual_content = (
                    f"{context}\n{chunk.page_content}" 
                    if context.strip() 
                    else chunk.page_content
                )
                
                contextual_chunks.append(Document(
                    page_content=contextual_content,
                    metadata=chunk.metadata.copy() if chunk.metadata else {}
                ))
                
                processed += 1
                if processed % 10 == 0:
                    logger.info(f"Processed {processed}/{total_chunks} chunks...")
        
        logger.info(f"Generated contextual summaries for {len(contextual_chunks)} chunks")
        return contextual_chunks
    
    def _group_chunks_by_document(self, chunks: List[Document]) -> dict:
        chunks_by_doc = defaultdict(list)
        for chunk in chunks:
            doc_id = chunk.metadata.get('article_id') or chunk.metadata.get('id', 'unknown')
            if doc_id and doc_id != 'unknown':
                chunks_by_doc[doc_id].append(chunk)
        return chunks_by_doc
    
    def _reconstruct_document(self, chunks: List[Document]) -> str:
        valid_chunks = [c for c in chunks if c and c.page_content]
        if not valid_chunks:
            return ""
        return '\n'.join(chunk.page_content for chunk in valid_chunks)
