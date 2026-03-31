import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_DIR = "data/chroma_db"


def get_retriever(k: int = 5, collection_name: str = "arxiv_papers"):
    """
    Devuelve un retriever LangChain listo para usar en la chain.
    k = número de chunks a recuperar por query.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


def search_with_sources(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Búsqueda semántica que devuelve chunks con su metadata completa.
    Útil para mostrar las fuentes en la interfaz.
    """
    retriever = get_retriever(k=k)
    docs = retriever.invoke(query)

    results = []
    for doc in docs:
        results.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "desconocida"),
            "topic": doc.metadata.get("topic", "general"),
            "chunk_index": doc.metadata.get("chunk_index", 0),
            "total_chunks": doc.metadata.get("total_chunks", 0),
        })
    return results


def get_unique_sources(query: str, k: int = 5) -> List[str]:
    """
    Devuelve solo los nombres de los papers usados como fuente.
    Para mostrar en la interfaz como lista de referencias.
    """
    results = search_with_sources(query, k=k)
    seen = set()
    sources = []
    for r in results:
        if r["source"] not in seen:
            seen.add(r["source"])
            sources.append(r["source"])
    return sources


if __name__ == "__main__":
    query = "What are the main components of a RAG system?"
    print(f"Query: {query}\n")
    results = search_with_sources(query, k=3)
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r['source']} — chunk {r['chunk_index']}")
        print(f"    {r['content'][:150]}...\n")