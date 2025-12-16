import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ragas import Dataset

logger = logging.getLogger(__name__)

try:
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_recall,
        context_precision,
        context_relevancy,
        answer_correctness,
        answer_similarity
    )
    from ragas import evaluate, Dataset
    from ragas.llms import LangchainLLMWrapper
    from tqdm import tqdm
    import pandas as pd
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    Dataset = None
    logger.warning("Ragas not available. Install with: pip install ragas tqdm")

try:
    from utils.resilience import initialize_llm_with_fallback
    RESILIENCE_AVAILABLE = True
except ImportError:
    RESILIENCE_AVAILABLE = False
    logger.warning("resilience module not available")


def _configure_ragas_llm():
    if not RAGAS_AVAILABLE or not RESILIENCE_AVAILABLE:
        return False
    
    try:
        llm = initialize_llm_with_fallback(temperature=0)
        if not llm:
            logger.warning("Failed to initialize LLM for Ragas. Ragas will use default LLM configuration.")
            return False
        
        ragas_llm = LangchainLLMWrapper(llm)
        
        faithfulness.llm = ragas_llm
        answer_relevancy.llm = ragas_llm
        answer_correctness.llm = ragas_llm
        context_precision.llm = ragas_llm
        context_relevancy.llm = ragas_llm
        context_recall.llm = ragas_llm
        
        logger.info("Configured Ragas metrics with explicit LLM (LLM-as-a-judge enabled)")
        return True
    except Exception as e:
        logger.warning(f"Failed to configure Ragas LLM: {e}. Ragas will use default LLM configuration.")
        return False


def create_ragas_dataset(
    rag_chain,
    eval_dataset: List[Dict[str, Any]],
    ticker_filter: Optional[str] = None,
    k: int = 3
):
    if not RAGAS_AVAILABLE:
        logger.error("Ragas is not installed. Cannot create evaluation dataset.")
        logger.error("Install with: pip install ragas tqdm")
        return None
    
    if not hasattr(rag_chain, 'query_for_evaluation'):
        logger.error("RAG chain must have query_for_evaluation method for evaluation")
        return None
    
    if not eval_dataset:
        logger.error("Evaluation dataset is empty")
        return None
    
    rag_dataset = []
    
    for row in tqdm(eval_dataset, desc="Creating Ragas dataset"):
        question = row.get("question", "")
        ground_truth = row.get("ground_truth", "")
        
        if not question:
            logger.warning("Skipping row with empty question")
            continue
        
        try:
            result = rag_chain.query_for_evaluation(
                question=question,
                ticker_filter=ticker_filter,
                k=k
            )
            
            answer = result.get("answer", "")
            if not answer:
                logger.warning(f"No answer generated for question: {question}")
                answer = ""
            
            contexts = result.get("context", [])
            if not isinstance(contexts, list):
                logger.warning(f"Contexts is not a list for question: {question}")
                contexts = []
            
            context_texts = []
            for ctx in contexts:
                if hasattr(ctx, 'page_content'):
                    context_texts.append(ctx.page_content)
                elif isinstance(ctx, str):
                    context_texts.append(ctx)
                elif isinstance(ctx, dict) and 'page_content' in ctx:
                    context_texts.append(ctx['page_content'])
                else:
                    logger.warning(f"Unexpected context type: {type(ctx)}")
            
            if not context_texts:
                logger.warning(f"No contexts extracted for question: {question}")
            
            rag_dataset.append({
                "question": question,
                "answer": answer,
                "contexts": context_texts,
                "ground_truths": [ground_truth] if ground_truth else [""]
            })
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}", exc_info=True)
            continue
    
    if not rag_dataset:
        logger.error("No valid data collected for Ragas dataset")
        return None
    
    try:
        rag_df = pd.DataFrame(rag_dataset)
        rag_eval_dataset = Dataset.from_pandas(rag_df)
        logger.info(f"Created Ragas dataset with {len(rag_dataset)} samples")
        return rag_eval_dataset
    except Exception as e:
        logger.error(f"Failed to create Ragas dataset: {e}", exc_info=True)
        return None


def evaluate_ragas_dataset(ragas_dataset) -> Optional[Dict[str, float]]:
    if not RAGAS_AVAILABLE:
        logger.error("Ragas is not installed. Cannot evaluate dataset.")
        return None
    
    if ragas_dataset is None:
        logger.error("Ragas dataset is None. Cannot evaluate.")
        return None
    
    _configure_ragas_llm()
    
    try:
        logger.info("Starting Ragas evaluation with 7 metrics (using LLM-as-a-judge)...")
        result = evaluate(
            ragas_dataset,
            metrics=[
                context_precision,
                faithfulness,
                answer_relevancy,
                context_recall,
                context_relevancy,
                answer_correctness,
                answer_similarity
            ],
        )
        
        logger.info("Ragas evaluation completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error during Ragas evaluation: {e}", exc_info=True)
        return None


def evaluate_rag_chain(
    rag_chain,
    eval_dataset: List[Dict[str, Any]],
    ticker_filter: Optional[str] = None,
    k: int = 3,
    save_dataset_path: Optional[str] = None
) -> Optional[Dict[str, float]]:
    if not eval_dataset:
        logger.error("Evaluation dataset is empty")
        return None
    
    logger.info(f"Starting RAG evaluation with {len(eval_dataset)} questions")
    
    ragas_dataset = create_ragas_dataset(
        rag_chain=rag_chain,
        eval_dataset=eval_dataset,
        ticker_filter=ticker_filter,
        k=k
    )
    
    if ragas_dataset is None:
        logger.error("Failed to create Ragas dataset")
        return None
    
    if save_dataset_path:
        try:
            ragas_dataset.to_pandas().to_csv(save_dataset_path, index=False)
            logger.info(f"Saved Ragas dataset to {save_dataset_path}")
        except Exception as e:
            logger.warning(f"Failed to save dataset to {save_dataset_path}: {e}")
    
    results = evaluate_ragas_dataset(ragas_dataset)
    
    if results:
        logger.info("\n=== Evaluation Results ===")
        for metric, score in results.items():
            logger.info(f"  {metric}: {score:.4f}")
    
    return results
