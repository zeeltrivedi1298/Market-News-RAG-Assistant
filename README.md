# Financial News Chat Application

A RAG (Retrieval-Augmented Generation) based chat application that allows users to ask questions about recent financial news and receive relevant summary responses.
Live link: https://842e5003c9cf9eb0d7.gradio.live/

## Features

- **Advanced RAG with Contextual Retrieval**: LLM-generated contextual summaries for chunks improve retrieval accuracy
- **Structured Citations**: Exact quoted text from sources with Pydantic models for verifiable answers
- **LCEL Pipeline**: Modern LangChain Expression Language for maintainable RAG chains
- **Semantic Search**: Sentence-transformers for embedding-based search
- **Vector Storage**: ChromaDB for efficient similarity search
- **Ticker Filtering**: Filter questions by specific stock tickers
- **Source Attribution**: Shows source articles with structured citations
- **Comprehensive Testing**: Unit, integration, and answer quality tests
- **Gradio UI**: Modern, interactive web interface

## Architecture

- **UI**: Gradio web interface
- **Backend**: Python with LangChain LCEL
- **Embeddings**: Sentence-transformers (HuggingFace)
- **Vector Store**: ChromaDB
- **LLM**: OpenAI API


## Project Structure

```
.
├── backend/
│   ├── gradio_main.py       # Gradio application (main entry point)
│   ├── config.py            # Configuration management
│   ├── core/                # Core RAG components
│   │   ├── __init__.py
│   │   ├── data_loader.py   # Data loading utilities
│   │   ├── vector_store.py  # Vector store with contextual retrieval
│   │   └── rag_chain.py      # LCEL-based RAG chain with citations
│   ├── models/              # Data models
│   │   ├── __init__.py
│   │   └── citation_models.py  # Pydantic models for structured citations
│   ├── utils/               # Utility functions
│   │   ├── __init__.py
│   │   ├── formatters.py    # Formatting utilities
│   │   ├── prompts.py       # Prompt templates
│   │   ├── processing_tracker.py  # Processing step tracking
│   │   ├── tracking_retriever.py  # Retriever wrapper for tracking
│   │   └── resilience.py    # Retry and fallback logic
│   ├── chunking/            # Document chunking strategies
│   │   ├── __init__.py
│   │   ├── factory.py
│   │   ├── token_chunker.py
│   │   ├── structure_chunker.py
│   │   └── semantic_chunker.py
│   ├── retrieval/           # Retrieval components
│   │   ├── __init__.py
│   │   ├── hybrid_retriever.py
│   │   └── reranker.py
│   ├── evaluation/          # Evaluation logic
│   │   ├── __init__.py
│   │   └── ragas_evaluation.py
│   ├── tests/               # Comprehensive test suite
│   │   ├── __init__.py
│   │   ├── test_data_loader.py
│   │   ├── test_vector_store.py
│   │   ├── test_rag_chain.py
│   │   ├── test_formatters.py
│   │   ├── test_citation_models.py
│   │   └── test_integration.py
│   ├── requirements.txt     # Python dependencies
│   ├── run.sh               # Quick start script
│   └── env.example          # Environment variables template
├── README.md                # This file
└── stock_news.json          # Financial news data
```

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key (optional if using local models)
```

5. Run the Gradio application:
```bash
python gradio_main.py
# Or use the run script:
./run.sh
```

The application will be available at `http://localhost:8000`

## Configuration

### Environment Variables

Create a `.env` file in the `backend` directory:

```env
# OpenAI API Key (required)
OPENAI_API_KEY=your_openai_api_key_here

# LLM Configuration
LLM_MODEL=gpt-3.5-turbo

# Embedding Configuration (free, local)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Vector Store Configuration
CHROMA_PERSIST_DIR=./chroma_db
```

## Usage

1. Start the Gradio application (see Setup Instructions above)
2. Open `http://localhost:8000` in your browser
3. Optionally select a ticker filter from the dropdown
4. Type your question in the input field
5. Click "Submit" or press Enter
6. View the answer along with sources and citations

### Example Questions

- "What are the latest developments with Apple?"
- "Tell me about recent AI stock trends"
- "What's happening with NVIDIA?"
- "Compare the performance of tech stocks"

## Testing

Comprehensive test suite focusing on answer quality:

### Run all tests:
```bash
cd backend
python -m pytest tests/ -v
```

### Run specific test suites:
```bash
# Unit tests
python -m unittest tests.test_data_loader
python -m unittest tests.test_citation_models
python -m unittest tests.test_formatters
python -m unittest tests.test_rag_chain

# Integration tests
python -m unittest tests.test_integration
python -m unittest tests.test_vector_store
```

### Test Coverage:
- **Unit Tests**: Data loading, citation models, component functionality
- **Integration Tests**: Full RAG pipeline with sample queries
- **Answer Quality Tests**: Answer relevance, citation accuracy, contextual retrieval effectiveness

## Application Interface

The Gradio interface provides:
- **Question Input**: Text area for entering questions
- **Ticker Filter**: Dropdown to filter by specific stock ticker (optional)
- **Processing Steps**: Visual display of all processing steps from question to answer
- **Answer Display**: Shows generated answer with formatting
- **Sources**: Lists all source articles used
- **Citations**: Shows exact quotes from sources with citations

## Technologies Used

- **Gradio**: Web UI framework for Python applications
- **LangChain LCEL**: Modern RAG pipeline with LangChain Expression Language
- **Pydantic**: Structured data validation for citations
- **ChromaDB**: Vector database
- **Sentence-Transformers**: Embedding models (HuggingFace) - free, local
- **OpenAI API**: LLM for answer generation and contextual retrieval

## Code Quality Features

- **Modular Architecture**: Separation of concerns (data loading, vector store, RAG chain)
- **Advanced RAG Techniques**: Contextual retrieval and structured citations
- **Comprehensive Testing**: Unit, integration, and answer quality tests
- **Type Safety**: Pydantic models for structured data
- **Error Handling**: Robust error handling throughout
- **Documentation**: Type hints and docstrings
- **Configuration Management**: Environment-based configuration

## Quality Focus

This implementation prioritizes:
- **Answer Quality**: Contextual retrieval and optimized prompts
- **Testing**: Comprehensive test suite (6 test files)
- **Simplicity**: Minimal, focused codebase
- **Verifiability**: Structured citations with exact quotes


