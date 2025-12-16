import os
import logging
import gradio as gr
from typing import Tuple, Optional

from config import Config
from core.data_loader import load_stock_news, get_tickers
from core.vector_store import VectorStoreManager
from core.rag_chain import RAGChain
from utils.processing_tracker import ProcessingStepsTracker

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

vector_store_manager = None
rag_chain = None
tickers = []

def initialize_rag_system() -> Tuple[bool, str]:
    global vector_store_manager, rag_chain, tickers
    
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        data_file = os.path.join(base_dir, "stock_news.json")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        logger.info("Loading stock news data...")
        documents = load_stock_news(data_file)
        logger.info(f"Loaded {len(documents)} documents")
        
        tickers_list = get_tickers(data_file)
        tickers.clear()
        tickers.extend(tickers_list)
        if tickers:
            logger.info(f"Available tickers: {', '.join(tickers)}")
        else:
            logger.debug("No tickers found in data")
        
        vector_store_manager = VectorStoreManager(use_contextual_retrieval=True)
        vector_store_manager.initialize_vector_store(documents, force_recreate=False)
        
        rag_chain = RAGChain(vector_store_manager)
        logger.info("RAG chain initialized successfully")
        
        return True, f"System initialized successfully. Loaded {len(documents)} documents."
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}", exc_info=True)
        vector_store_manager = None
        rag_chain = None
        tickers.clear()
        return False, f"Error initializing system: {str(e)}"

def query_rag(question: str, ticker: Optional[str], k: int = 3):
    global rag_chain, tickers
    
    tracker = ProcessingStepsTracker()
    
    if question is None:
        question = ""
    
    logger.info(f"Received query: {question[:50] if question else 'None'}...")
    
    step1 = tracker.start_step(
        "Question Validation",
        "Validating and normalizing the user's question",
        {"question": (question[:100] + "..." if len(question) > 100 else question) if question else "None"}
    )
    
    if not question or not question.strip():
        tracker.error_step(step1, "Question is empty")
        return "Please enter a question.", "", "", tracker.to_markdown()
    
    if rag_chain is None:
        tracker.error_step(step1, "RAG chain not initialized")
        logger.error("RAG chain is None")
        return "Error: RAG system not initialized. Please check the logs.", "", "", tracker.to_markdown()
    
    question_normalized = question.strip()
    tracker.complete_step(step1, {"normalized_question": question_normalized})
    
    step2 = tracker.start_step(
        "Ticker Filter Processing",
        "Processing and validating ticker filter",
        {"selected_ticker": ticker or "None"}
    )
    
    ticker_filter = _normalize_ticker_filter(ticker, tickers)
    if isinstance(ticker_filter, str) and ticker_filter.startswith("Error:"):
        tracker.error_step(step2, ticker_filter)
        return ticker_filter, "", "", tracker.to_markdown()
    
    tracker.complete_step(step2, {
        "ticker_filter": ticker_filter or "None",
        "filter_applied": ticker_filter is not None
    })
    
    step3 = tracker.start_step(
        "Retrieval Parameters",
        "Setting up document retrieval parameters",
        {"k": k}
    )
    
    k = max(1, min(20, k if isinstance(k, int) else 3))
    tracker.complete_step(step3, {"final_k": k})
    
    try:
        if ticker_filter:
            logger.info(f"Processing query with k={k}, ticker_filter={ticker_filter}")
        else:
            logger.debug(f"Processing query with k={k}, no ticker filter")
        
        result = rag_chain.query(
            question=question_normalized,
            ticker_filter=ticker_filter,
            k=k,
            tracker=tracker
        )
        
        answer = result.get("answer", "No answer generated.")
        sources = result.get("sources", [])
        citations = result.get("citations", [])
        
        no_information_indicators = [
            "i don't have information",
            "don't have information",
            "couldn't generate",
            "no information about"
        ]
        answer_lower = answer.lower()
        has_no_information = any(indicator in answer_lower for indicator in no_information_indicators)
        
        if has_no_information:
            sources_text = "## Sources\n\n*No relevant sources found for this query.*\n"
            citations_text = "## Citations\n\n*No citations available as no relevant information was found.*\n"
        else:
            sources_text = _format_sources(sources)
            citations_text = _format_citations(citations)
        
        logger.info("Query completed successfully")
        return answer, sources_text, citations_text, tracker.to_markdown()
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return f"Error: {e}", "", "", tracker.to_markdown()
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return f"An error occurred: {e}", "", "", tracker.to_markdown()

def _normalize_ticker_filter(ticker: Optional[str], valid_tickers: list) -> Optional[str]:
    if not ticker or (isinstance(ticker, str) and ticker.strip().upper() == "NONE"):
        return None
    
    if not isinstance(ticker, str):
        return f"Error: Invalid ticker type. Expected string, got {type(ticker).__name__}."
    
    if not valid_tickers or len(valid_tickers) == 0:
        logger.warning("No valid tickers available for filtering")
        return None
    
    ticker_filter = ticker.strip().upper()
    if ticker_filter not in valid_tickers:
        return f"Error: Invalid ticker '{ticker_filter}'. Please select a valid ticker."
    
    return ticker_filter

def _format_sources(sources: list) -> str:
    if not sources:
        return "## Sources\n\nNo sources found.\n"
    
    def _extract_publisher(link: str) -> str:
        if not link:
            return ""
        if "yahoo.com" in link.lower():
            return "Yahoo Finance"
        elif "insidermonkey.com" in link.lower():
            return "Insider Monkey"
        elif "motleyfool.com" in link.lower():
            return "The Motley Fool"
        elif "zacks.com" in link.lower():
            return "Zacks Investment Research"
        return ""
    
    lines = ["## Sources\n"]
    lines.append("*All documents retrieved and used to generate the answer:*\n")
    
    for i, source in enumerate(sources, 1):
        title = source.get("title", "Unknown")
        ticker = source.get("ticker", "Unknown")
        link = source.get("link", "")
        publisher = _extract_publisher(link)
        publisher_text = f" — {publisher}" if publisher else ""
        link_text = f" [Link]({link})" if link else ""
        lines.append(f"{i}. **{title}** ({ticker}){publisher_text}{link_text}")
    
    return "\n".join(lines) + "\n"

def _format_citations(citations: list) -> str:
    if not citations:
        return "## Citations\n\nNo citations found.\n"
    
    lines = ["## Citations\n"]
    lines.append("*Specific quoted excerpts from sources that support the answer:*\n")
    
    for i, citation in enumerate(citations, 1):
        title = citation.get("title", "Unknown")
        ticker = citation.get("ticker", "Unknown")
        quotes = citation.get("quotes", "").strip()
        
        citation_line = f"{i}. **{title}** ({ticker})"
        
        if quotes:
            quote_preview = quotes[:200] + "..." if len(quotes) > 200 else quotes
            citation_line += f"\n   > *\"{quote_preview}\"*"
        
        lines.append(citation_line)
        lines.append("")  # Add spacing between citations
    
    return "\n".join(lines) + "\n"

def create_gradio_interface():
    global tickers
    
    success, message = initialize_rag_system()
    if not success:
        logger.error(f"Failed to initialize: {message}")
    
    ticker_options = ["None"] + sorted(tickers) if tickers else ["None"]
    
    with gr.Blocks(title="Financial News Chat") as demo:
        gr.Markdown(
            """
            # 📈 Financial News Chat Application
            
            Ask questions about recent financial news. You can optionally filter by stock ticker.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., What are the latest developments with Apple?",
                    lines=3
                )
                
                ticker_dropdown = gr.Dropdown(
                    choices=ticker_options,
                    value="None",
                    label="Filter by Ticker (Optional)",
                    info="Select a stock ticker to filter results"
                )
                
                submit_btn = gr.Button("Submit", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### Available Tickers")
                ticker_list = gr.Markdown(
                    value="\n".join([f"- {t}" for t in sorted(tickers)]) if tickers else "Loading..."
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Your Answer")
                answer_output = gr.Markdown(
                    value="Enter a question above and click Submit."
                )
        
        with gr.Row():
            with gr.Column():
                sources_output = gr.Markdown(
                    label="Sources",
                    value=""
                )
            
            with gr.Column():
                citations_output = gr.Markdown(
                    label="Citations",
                    value=""
                )
        
        with gr.Row():
            with gr.Column():
                processing_steps_output = gr.HTML(
                    value="""
                    <div style="
                        padding: 20px;
                        text-align: center;
                        color: #6B7280;
                        background-color: #F9FAFB;
                        border-radius: 8px;
                        border: 2px dashed #D1D5DB;
                    ">
                        <div style="font-size: 48px; margin-bottom: 10px;">⚙️</div>
                        <div style="font-size: 16px; font-weight: 600; margin-bottom: 4px;">Processing Steps</div>
                        <div style="font-size: 13px;">Steps will appear here after you submit a question</div>
                    </div>
                    """
                )
        
        submit_btn.click(
            fn=query_rag,
            inputs=[question_input, ticker_dropdown],
            outputs=[answer_output, sources_output, citations_output, processing_steps_output]
        )
        
        question_input.submit(
            fn=query_rag,
            inputs=[question_input, ticker_dropdown],
            outputs=[answer_output, sources_output, citations_output, processing_steps_output]
        )
        
        gr.Examples(
            examples=[
                ["What are the latest developments with Apple?", "None"],
                ["Tell me about recent AI stock trends", "None"],
                ["What's happening with NVIDIA?", "None"],
                ["Compare the performance of tech stocks", "None"],
            ],
            inputs=[question_input, ticker_dropdown]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="127.0.0.1" if Config.HOST == "0.0.0.0" else Config.HOST,
        server_port=Config.PORT,
        share=Config.GRADIO_SHARE,
        theme=gr.themes.Soft()
    )

