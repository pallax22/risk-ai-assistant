import arxiv
import os
import logging
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PAPERS_DIR = Path("data/papers")


def search_papers(
    query: str,
    max_results: int = 10,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance
) -> List[arxiv.Result]:
    """
    Busca papers en ArXiv dado un tema.
    Devuelve lista de resultados con metadata.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_by
    )
    results = list(client.results(search))
    logger.info(f"Encontrados {len(results)} papers para: '{query}'")
    return results


def download_papers(
    results: List[arxiv.Result],
    topic_folder: Optional[str] = None
) -> List[Path]:
    """
    Descarga los PDFs de los papers encontrados.
    Los guarda en data/papers/{topic_folder}/
    Devuelve lista de rutas a los PDFs descargados.
    """
    folder = PAPERS_DIR / (topic_folder or "general")
    folder.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for paper in results:
        # Nombre de archivo limpio basado en el ID de ArXiv
        paper_id = paper.entry_id.split("/")[-1]
        pdf_path = folder / f"{paper_id}.pdf"

        if pdf_path.exists():
            logger.info(f"Ya existe: {pdf_path.name}")
            downloaded.append(pdf_path)
            continue

        try:
            paper.download_pdf(dirpath=str(folder), filename=f"{paper_id}.pdf")
            downloaded.append(pdf_path)
            logger.info(f"Descargado: {paper.title[:60]}...")
        except Exception as e:
            logger.error(f"Error descargando {paper_id}: {e}")

    logger.info(f"PDFs disponibles: {len(downloaded)}/{len(results)}")
    return downloaded


def get_paper_metadata(results: List[arxiv.Result]) -> List[dict]:
    """
    Extrae metadata clave de cada paper.
    Útil para mostrar fuentes en la interfaz.
    """
    metadata = []
    for paper in results:
        metadata.append({
            "id": paper.entry_id.split("/")[-1],
            "title": paper.title,
            "authors": [a.name for a in paper.authors[:3]],  # Primeros 3 autores
            "published": paper.published.strftime("%Y-%m-%d"),
            "abstract": paper.summary[:300] + "...",
            "url": paper.entry_id,
            "pdf_url": paper.pdf_url,
            "categories": paper.categories
        })
    return metadata


if __name__ == "__main__":
    # Test rápido — ejecuta con: python src/arxiv_client.py
    print("Buscando papers sobre RAG...\n")
    results = search_papers("retrieval augmented generation LLM", max_results=3)
    
    for i, paper in enumerate(results, 1):
        print(f"{i}. {paper.title}")
        print(f"   Autores: {', '.join(a.name for a in paper.authors[:2])}")
        print(f"   Publicado: {paper.published.strftime('%Y-%m-%d')}")
        print()
    
    print("Descargando PDFs...")
    paths = download_papers(results, topic_folder="test")
    print(f"\nDescargados en: {paths}")