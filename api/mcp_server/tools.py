from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.retrievers import ArxivRetriever


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
