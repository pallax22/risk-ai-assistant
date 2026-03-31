import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.retriever import get_retriever, get_unique_sources

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert AI research assistant specializing in machine learning, \
NLP, and artificial intelligence papers from ArXiv.

Your role is to help researchers and practitioners understand complex AI papers \
by providing clear, accurate, and well-structured answers.

When answering:
1. Base your response EXCLUSIVELY on the provided context from the papers.
2. Structure your answer with: Summary, Key findings, Methodology (if relevant), and Limitations.
3. Always cite which paper(s) your information comes from.
4. If the context doesn't contain enough information, say so clearly.
5. Use precise technical language appropriate for an ML audience.
6. If comparing multiple approaches, use a structured format.

Context from retrieved papers:
{context}

Remember: cite your sources and be precise."""

HUMAN_PROMPT = "{question}"


def format_docs(docs) -> str:
    """Formatea los documentos recuperados en un string de contexto."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Source {i} - {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain():
    """
    Construye la chain RAG completa:
    query → retriever → prompt → LLM → respuesta
    """
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT)
    ])

    retriever = get_retriever(k=5)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def ask(question: str) -> Dict[str, Any]:
    """
    Función principal del asistente.
    Devuelve la respuesta + las fuentes usadas.
    """
    chain = build_rag_chain()
    sources = get_unique_sources(question, k=5)

    logger.info(f"Procesando: {question[:60]}...")
    response = chain.invoke(question)

    return {
        "answer": response,
        "sources": sources,
        "question": question
    }


if __name__ == "__main__":
    questions = [
        "What are the main components of a RAG system and how do they interact?",
        "What evaluation metrics are used to measure RAG performance?",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print('='*60)
        result = ask(q)
        print(f"\n{result['answer']}")
        print(f"\nSources used: {', '.join(result['sources'])}")