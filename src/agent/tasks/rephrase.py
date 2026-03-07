from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

contextualize_q_system_prompt = """
You are an expert in question formulation for molecular biology.
Your task is to analyze the conversation history and the user's latest query to create a standalone version of the question.

IMPORTANT:
- If the user's question is NOT in English, translate it to English for this step.
- Internal Search Optimization: This English translation is strictly for optimizing vector search and keyword matching in the Reactome and UniProt databases.
- The standalone question should be clear, concise, and scientifically accurate.
- Do NOT answer the question. Only return the reformulated English question.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}"),
    ]
)


def create_rephrase_chain(llm: BaseChatModel) -> Runnable:
    return (contextualize_q_prompt | llm | StrOutputParser()).with_config(
        run_name="rephrase_question"
    )
