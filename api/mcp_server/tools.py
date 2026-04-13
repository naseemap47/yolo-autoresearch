from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.retrievers import ArxivRetriever
from ultralytics import YOLO


async def search_web_links(query: str) -> list | None:
    try:
        search = DuckDuckGoSearchResults(output_format="list", num_results=4)
        search_result = search.invoke(query)
        return search_result
    except Exception as e:
        print(f"Error Search the query in web using DuckDuckGo: {e}")
        raise

async def read_url(url: str) -> str:
    try:
        loader = WebBaseLoader(url)
        result_web_read = loader.load()
        return result_web_read[0].page_content
    except Exception as e:
        print(f"Error reading {url} web: {e}")

async def search_wiki(query: str):
    try:
        retriever = WikipediaRetriever()
        docs = retriever.invoke(query)
        return docs
    except Exception as e:
        print(f"Error Search the query in Wikipedia: {e}")
        raise

async def search_arxiv(query: str):
    try:
        retriever = ArxivRetriever(
            load_max_docs=3,
            get_full_documents=True,
        )
        docs = retriever.invoke(query)
        return docs
    except Exception as e:
        print(f"Error Search the query in ArXiv: {e}")
        raise

async def train_model(
        model_name: str, data_path: str, epochs: int,
        patience: int, batch: int | float,
        imgsz: int, device: int | str | list,
        name: str, optimizer: str, single_cls: bool,
        lr0: float, lrf: float, momentum: float, weight_decay: float,
):
    model = YOLO(model_name)
    result =model.train(
        data=data_path, epochs=epochs,
        patience=patience, batch=batch,
        imgsz=imgsz, device=device,
        name=name, optimizer=optimizer,
        single_cls=single_cls, lr0=lr0,
        lrf=lrf, momentum=momentum,
        weight_decay=weight_decay,
    )
    return result

async def evaluate_model(model_path):
    model = YOLO(model_path)
    results = model.val()
    return results
