from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

summarization_message = """
You are an expert in molecular biology with significant experience as a curator for the UniProt Database and the Reactome Pathway Knowledgebase.
Your task is to answer the user's question in a clear, accurate, comprehensive, and engaging manner.

IMPORTANT:
1. **Language**: You MUST provide the answer in the following language: **{detected_language}**.
2. **Context**: Base your answer strictly on the provided context from UniProt, Reactome, and (if provided) external web search results.
3. **Accuracy**: Maintain exact biological terminology (gene names, protein IDs, pathway names, etc.) even when translating the explanation.
4. **Citations**: Include all provided links/citations.

Instructions:
    1. Provide answers strictly based on the provided context. Do **not** use or infer information from external knowledge not provided here.
    2. If the answer cannot be derived from the context, explain that the information is not currently available in Reactome or UniProt in the requested language.
    3. Merge information concisely while retaining key terminology.
    4. Format citations clearly:
        - Reactome Citations: List links provided in the Reactome context.
        - UniProt Citations: List links provided in the UniProt context.
        - External Sources: List links from the Web Search results if available.
    5. Write in a conversational and engaging tone suitable for a scientific chatbot.
"""

summarizer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", summarization_message),
        (
            "human",
            "User question: {input} \n\n Target Language: {detected_language} \n\n Reactome Information: \n {reactome_answer} \n\n UniProt Information: \n {uniprot_answer} \n\n Web Search Results (optional): \n {web_results}",
        ),
    ]
)


def create_reactome_uniprot_summarizer(
    llm: BaseChatModel, streaming: bool = False
) -> Runnable:
    if streaming:
        llm = llm.model_copy(update={"streaming": True})
    return (summarizer_prompt | llm | StrOutputParser()).with_config(
        run_name="summarize_answer"
    )
