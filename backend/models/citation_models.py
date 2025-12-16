from pydantic import BaseModel, Field
from typing import List

class Citation(BaseModel):
    id: str = Field(description="The string ID of a SPECIFIC context article which justifies the answer.")
    source: str = Field(description="The source of the SPECIFIC context article which justifies the answer.")
    title: str = Field(description="The title of the SPECIFIC context article which justifies the answer.")
    ticker: str = Field(description="The stock ticker associated with this citation.")
    page: int = Field(default=1, description="The page number of the SPECIFIC context article which justifies the answer.")
    quotes: str = Field(description="The VERBATIM sentences from the SPECIFIC context article that are used to generate the answer. Should be exact sentences from context article without missing words.")

class QuotedCitations(BaseModel):
    citations: List[Citation] = Field(description="Citations (can be multiple) from the given context articles that justify the answer.")
