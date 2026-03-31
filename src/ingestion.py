import os
import logging
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_DIR = "data/chroma_db"
PAPERS_DIR = Path("data/papers")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extrae todo el texto de un PDF usando PyMuPDF.
    Más robusto que pdfplumber para papers científicos.
    """
    doc = fitz.open(str(pdf_path))
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    # Limpieza básica
    text = " ".join(text.split())
    logger.info(f"Extraídos {len(text)} caracteres de {pdf_path.name}")
    return text


def load_papers_from_folder(folder: Optional[Path] = None) -> List[dict]:
    """
    Carga todos los PDFs de una carpeta y extrae su texto.
    Devuelve lista de dicts con texto + metadata.
    """
    folder = folder or PAPERS_DIR
    papers = []

    pdf_files = list(folder.rglob("*.pdf"))
    logger.info(f"Encontrados {len(pdf_files)} PDFs en {folder}")

    for pdf_path in pdf_files:
        try:
            text = extract_text_from_pdf(pdf_path)
            if len(text) < 100:
                logger.warning(f"PDF vacío o sin texto: {pdf_path.name}")
                continue

            papers.append({
                "text": text,
                "metadata": {
                    "source": pdf_path.name,
                    "path": str(pdf_path),
                    "topic": pdf_path.parent.name
                }
            })
        except Exception as e:
            logger.error(f"Error procesando {pdf_path.name}: {e}")

    logger.info(f"Papers cargados correctamente: {len(papers)}")
    return papers


def split_into_chunks(papers: List[dict]) -> List[dict]:
    """
    Divide cada paper en chunks solapados.
    chunk_size=800 y overlap=150 es el balance óptimo
    para papers científicos: suficiente contexto sin perder precisión.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for paper in papers:
        splits = splitter.split_text(paper["text"])
        for i, chunk in enumerate(splits):
            chunks.append({
                "text": chunk,
                "metadata": {
                    **paper["metadata"],
                    "chunk_index": i,
                    "total_chunks": len(splits)
                }
            })

    logger.info(f"Total chunks generados: {len(chunks)}")
    return chunks


def build_vector_store(chunks: List[dict], collection_name: str = "arxiv_papers") -> Chroma:
    """
    Genera embeddings con OpenAI y los guarda en ChromaDB.
    La primera vez tarda según el número de chunks.
    Las siguientes ejecuciones cargan desde disco (rápido).
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    logger.info(f"Generando embeddings para {len(texts)} chunks...")
    logger.info("Esto puede tardar 1-2 minutos la primera vez...")

    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=collection_name,
        persist_directory=CHROMA_DIR
    )

    logger.info(f"ChromaDB guardado en: {CHROMA_DIR}")
    logger.info(f"Total vectores indexados: {vector_store._collection.count()}")
    return vector_store


def load_vector_store(collection_name: str = "arxiv_papers") -> Chroma:
    """
    Carga un vector store ya existente desde disco.
    Usa esto en lugar de build_vector_store() cuando ya tienes
    los embeddings generados.
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

    count = vector_store._collection.count()
    logger.info(f"Vector store cargado: {count} vectores")
    return vector_store


def run_full_pipeline(topic_folder: Optional[str] = None) -> Chroma:
    """
    Pipeline completo: PDF → texto → chunks → embeddings → ChromaDB.
    Llama a esta función desde otros módulos.
    """
    folder = PAPERS_DIR / topic_folder if topic_folder else PAPERS_DIR

    papers = load_papers_from_folder(folder)
    if not papers:
        raise ValueError(f"No se encontraron PDFs válidos en {folder}")

    chunks = split_into_chunks(papers)
    vector_store = build_vector_store(chunks)
    return vector_store


if __name__ == "__main__":
    import sys

    # Comprueba si ya existe la DB para no regenerar embeddings
    chroma_path = Path(CHROMA_DIR)
    if chroma_path.exists() and any(chroma_path.iterdir()):
        print("Vector store ya existe. Cargando desde disco...")
        vs = load_vector_store()
    else:
        print("Construyendo vector store desde cero...")
        folder_arg = sys.argv[1] if len(sys.argv) > 1 else None
        vs = run_full_pipeline(folder_arg)

    # Test de búsqueda semántica
    print("\nTest de búsqueda semántica:")
    print("-" * 40)
    query = "How does retrieval augmented generation improve LLM accuracy?"
    results = vs.similarity_search(query, k=3)

    for i, doc in enumerate(results, 1):
        print(f"\nResultado {i}:")
        print(f"  Fuente: {doc.metadata.get('source', 'desconocida')}")
        print(f"  Chunk:  {doc.metadata.get('chunk_index', '?')} / {doc.metadata.get('total_chunks', '?')}")
        print(f"  Texto:  {doc.page_content[:200]}...")