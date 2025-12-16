from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = """You are an expert financial news analyst. Answer the question using ONLY the provided context from recent financial news articles.

Guidelines:
- Use only information from the provided context
- If the answer is not in the context, say "I don't have information about that in the provided context"
- Be specific and mention relevant stock tickers when applicable
- Provide clear, well-structured answers
- Cite specific details from the context
- Do NOT include star ratings (★), rating symbols, or any visual rating indicators in your answer

Question: {question}

Context:
{context}

Answer:"""

CITATIONS_PROMPT = """You are an expert at analyzing answers and extracting exact citations from source articles.

Task: Analyze the generated answer and extract VERBATIM quotes from the context articles that justify each part of the answer.

Requirements:
- Extract exact sentences/phrases from context articles (do not paraphrase)
- Match citations to articles by their ID, source, title, and ticker
- Only cite articles that were actually used in generating the answer
- Include the full quote that supports the answer

Question: {question}

Context Articles:
{context}

Generated Answer:
{answer}

Extract citations with exact quotes:"""

def get_rag_prompt_template():
    return ChatPromptTemplate.from_template(RAG_PROMPT)

def get_citations_prompt_template():
    return ChatPromptTemplate.from_template(CITATIONS_PROMPT)
