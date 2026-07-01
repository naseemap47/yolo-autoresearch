"""
RAG (Retrieval-Augmented Generation) module for the YOLO Auto-Research Agent.

Ingests Ultralytics documentation from a pre-scraped text file into a persistent
ChromaDB vector store. At planning time, the agent queries this store to retrieve
the most relevant documentation chunks, which are then injected into the LLM
planner's prompt as grounding context.
"""

import os
import logging
from typing import List
import requests
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load Ultralytics Docs
# ---------------------------------------------------------------------------

def load_ultralytics_docs():
    urls = [
        "https://docs.ultralytics.com/modes/train#introduction", 
        "https://www.ultralytics.com/glossary/epoch#the-role-of-epochs-in-optimization",
        "https://www.ultralytics.com/glossary/overfitting",
        "https://www.ultralytics.com/glossary/batch-size",
        "https://www.ultralytics.com/glossary/accuracy",
        "https://www.ultralytics.com/glossary/learning-rate",
        "https://www.ultralytics.com/glossary/data-augmentation",
        "https://www.ultralytics.com/glossary/mixed-precision",
        "https://www.ultralytics.com/glossary/transfer-learning",
        "https://www.ultralytics.com/glossary/adam-optimizer",
        "https://www.ultralytics.com/glossary/regularization",
        "https://www.ultralytics.com/glossary/loss-function#the-role-of-loss-in-model-training",
        "https://www.ultralytics.com/glossary/bounding-box",
        "https://docs.ultralytics.com/guides/yolo-data-augmentation",
        "https://docs.ultralytics.com/guides/hyperparameter-tuning",
        "https://docs.ultralytics.com/usage/cfg",
        "https://github.com/ultralytics/ultralytics/issues/7749",
        "https://github.com/orgs/ultralytics/discussions/9536",
        "https://medium.com/internet-of-technology/yolov8-best-practices-for-training-cdb6eacf7e4f",
        "https://docs.ultralytics.com/guides/model-training-tips#how-can-i-use-pretrained-weights-to-speed-up-training-in-yolo26",
        "https://docs.ultralytics.com/yolov5/tutorials/tips-for-best-training-results#dataset",
        "https://clarion.ai/10-tips-to-train-deep-learning-model-using-yolo/",
        "https://keylabs.ai/blog/training-yolov8-models-tips-for-success/",
        "https://github.com/orgs/ultralytics/discussions/2799",
        "https://github.com/orgs/ultralytics/discussions/24292",
        "https://github.com/ultralytics/yolov5/discussions/9198",
        "https://sodevelopment.medium.com/top-5-tips-for-training-yolo-mastering-object-detection-with-confidence-463e54b2a7a7",
        "https://docs.ultralytics.com/yolov5/tutorials/transfer-learning-with-frozen-layers#before-you-start",
        "https://docs.ultralytics.com/guides/finetuning-guide#fine-tuning-vs-training-from-scratch",
        "https://www.ultralytics.com/glossary/regularization#core-concepts-and-techniques",
        
    ]

    scraped_data = []

    for url in urls:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Target the main article body on MKDocs/Material-based documentation
            main_content = soup.find('article') or soup.find('main')
            if main_content:
                # Clean up the text
                text = main_content.get_text(separator='\n', strip=True)
                # Remove excessive newlines
                text = re.sub(r'\n+', '\n', text)
                scraped_data.append(text)

    # Save raw text chunks for the next step
    with open("finetune/ultralytics_raw.txt", "w", encoding="utf-8") as f:
        f.write("\n---DOCUMENT SPLIT---\n".join(scraped_data))

# ---------------------------------------------------------------------------
# Text splitting helpers
# ---------------------------------------------------------------------------

def _split_into_chunks(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
    """
    Split a long text into overlapping chunks of approximately `chunk_size`
    characters. Tries to break at newline boundaries where possible.
    """
    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        # Try to break at a newline to avoid cutting mid-sentence
        if end < length:
            newline_pos = text.rfind("\n", start, end)
            if newline_pos > start + chunk_size // 2:
                end = newline_pos

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - chunk_overlap if end - chunk_overlap > start else end

    return chunks


def _load_and_split_docs(docs_path: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Load the scraped Ultralytics raw text file, split by the document boundary
    marker (---DOCUMENT SPLIT---), then further chunk each document section.
    Returns a flat list of text chunks.
    """
    if not os.path.exists(docs_path):
        # raise FileNotFoundError(
        #     f"Docs file not found at '{docs_path}'. "
        #     "Make sure the finetune/ultralytics_raw.txt file exists. "
        #     "Run the scraping cell in finetune/finetune.ipynb first."
        # )
        logger.info(f"Docs file not found at '{docs_path}'. "
            "Scraping Ultralytics docs now.")
        load_ultralytics_docs()
        logger.info(f"Successfully scraped docs to '{docs_path}'")

    with open(docs_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Split by the boundary marker written by the scraping notebook
    sections = [s.strip() for s in raw.split("---DOCUMENT SPLIT---") if s.strip()]
    logger.info(f"Loaded {len(sections)} document sections from '{docs_path}'")

    all_chunks: List[str] = []
    for section in sections:
        chunks = _split_into_chunks(section, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)

    logger.info(f"Split into {len(all_chunks)} total chunks (chunk_size={chunk_size})")
    return all_chunks


# ---------------------------------------------------------------------------
# Vector store construction / loading
# ---------------------------------------------------------------------------

def build_vectorstore(
    docs_path: str,
    persist_dir: str,
    embedding_model: str = "nomic-embed-text",
    chunk_size: int = 800,
    chunk_overlap: int = 100,
):
    """
    Ingest the scraped Ultralytics docs, embed them, and persist a ChromaDB
    collection to `persist_dir`.  Returns the Chroma vector store object.
    """
    try:
        import chromadb
        from langchain_ollama import OllamaEmbeddings
        from langchain_community.vectorstores import Chroma
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}. "
            "Run: uv add chromadb  (or pip install chromadb)"
        ) from e

    print(f"[RAG] Building vectorstore from '{docs_path}' …")
    chunks = _load_and_split_docs(docs_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    embeddings = OllamaEmbeddings(model=embedding_model)

    # Chroma will create the persist_dir if it doesn't exist
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="ultralytics_docs",
    )

    print(f"[RAG] Vectorstore built with {len(chunks)} chunks → saved to '{persist_dir}'")
    return vectorstore


def load_vectorstore(
    persist_dir: str,
    embedding_model: str = "nomic-embed-text",
):
    """
    Load an existing ChromaDB vector store from `persist_dir` without
    re-ingesting or re-embedding the source documents.
    """
    try:
        from langchain_ollama import OllamaEmbeddings
        from langchain_chroma import Chroma
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}. "
            "Run: uv add chromadb  (or pip install chromadb)"
        ) from e

    print(f"[RAG] Loading existing vectorstore from '{persist_dir}' …")
    embeddings = OllamaEmbeddings(model=embedding_model)
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_name="ultralytics_docs",
    )
    print("[RAG] Vectorstore loaded.")
    return vectorstore


def get_or_build_vectorstore(
    docs_path: str,
    persist_dir: str,
    embedding_model: str = "nomic-embed-text",
    chunk_size: int = 800,
    chunk_overlap: int = 100,
):
    """
    Convenience helper: loads an existing store if `persist_dir` already
    contains a ChromaDB database, otherwise builds one from scratch.
    """
    # ChromaDB creates a `chroma.sqlite3` file inside persist_dir
    db_file = os.path.join(persist_dir, "chroma.sqlite3")
    if os.path.exists(db_file):
        return load_vectorstore(persist_dir=persist_dir, embedding_model=embedding_model)
    else:
        return build_vectorstore(
            docs_path=docs_path,
            persist_dir=persist_dir,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


# ---------------------------------------------------------------------------
# Retriever wrapper
# ---------------------------------------------------------------------------

class UltralyticsRetriever:
    """
    Thin wrapper around a Chroma vector store that provides a `.query()` method
    returning a formatted string of retrieved documentation chunks, ready to be
    injected into the LLM planner prompt as `{context}`.
    """

    def __init__(self, vectorstore, top_k: int = 4):
        self._vs = vectorstore
        self._top_k = top_k

    def query(self, question: str) -> str:
        """
        Retrieve the `top_k` most relevant documentation chunks for `question`
        and return them as a single formatted string.
        """
        try:
            docs = self._vs.similarity_search(question, k=self._top_k)
            if not docs:
                return "(No relevant documentation found.)"

            parts = []
            for i, doc in enumerate(docs, start=1):
                parts.append(f"[Doc {i}]\n{doc.page_content.strip()}")

            return "\n\n".join(parts)
        except Exception as e:
            logger.warning(f"[RAG] Retrieval failed: {e}")
            return "(Documentation retrieval unavailable.)"
