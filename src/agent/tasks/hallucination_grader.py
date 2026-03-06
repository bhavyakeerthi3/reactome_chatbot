from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

hallucination_grader_message = """\
You are an expert scientific fact-checker with deep knowledge of molecular biology, \
the Reactome Pathway Knowledgebase, and the UniProt Knowledgebase.

Your task is to assess whether a given LLM-generated answer is **grounded** in the \
provided source documents. An answer is grounded if every factual claim it makes can \
be directly traced to the retrieved context below.

Respond with a binary output:
    - Yes: Every factual claim in the answer is supported by the retrieved documents.
    - No: The answer contains at least one claim that is NOT supported by the \
retrieved documents (i.e., hallucinated or fabricated).

Do NOT penalise an answer for being incomplete — only penalise unsupported claims.
Do NOT use any external knowledge; judge only against the provided documents.
"""

hallucination_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", hallucination_grader_message),
        (
            "human",
            "Retrieved documents:\n\n{documents}\n\nLLM generation:\n\n{generation}",
        ),
    ]
)


class HallucinationGrade(BaseModel):
    binary_score: str = Field(
        description=(
            "Indicates whether the answer is grounded in the retrieved documents. "
            "'Yes' means fully grounded, 'No' means at least one hallucinated claim."
        )
    )
    reason: str = Field(
        default="",
        description=(
            "If binary_score is 'No', briefly state which claim is not supported. "
            "Leave empty when fully grounded."
        ),
    )


def format_documents(documents: list[Document]) -> str:
    """Concatenate document page content for prompt injection."""
    return "\n\n".join(doc.page_content for doc in documents)


def create_hallucination_grader(llm: BaseChatModel) -> Runnable:
    return hallucination_grader_prompt | llm.with_structured_output(HallucinationGrade)
