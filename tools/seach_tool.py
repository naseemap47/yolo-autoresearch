from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.retrievers import ArxivRetriever
from langchain.tools import tool


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

@tool
async def internet_search(query: str):
    """
    Web Search Tool to search and read docs, articles and News from web.
    While using this tool use wiki_search tool for searching in wikipedia and use this web_search tool for searching in other web sources.
    
    Args:
        query: The query to search for (eg. "Who is Elon Musk ?", "History of Japan", "What is the latest news about AI ?")
    
    Returns:
        extracted text
    """
    web_results = await search_web_links(query)
    if len(web_results) == 0:
        return "No results found"
    else:
        text = ""
        for web_result in web_results:
            text += await read_url(web_result['link'])
        return text

@tool
async def wiki_search(query: str):
    """
    Wikipedia search tool to search and and get the details about the query.

    Args:
        query: The query to search for (eg. "Who is Elon Musk ?", "Japan history")
    
    Returns:
        extracted text from wikipedia about the query
    """
    docs = await search_wiki(query)
    if len(docs) == 0:
        return "No results found"
    else:
        text = ""
        for doc in docs:
            text += doc.page_content
        return text

@tool
async def arxiv_search(query: str):
    """
    ArXiv search tool to search about research papers from ArXiv, and get the details about the query.

    Args:
        query: The query to search for (eg. "Expain me about 1605.08386 research paper.", "explain me about Transformers from Ai model architecture")
    
    Returns:
        extracted text from ArXiv about the query
    """
    docs = await search_arxiv(query)
    if len(docs) == 0:
        return "No results found"
    else:
        text = ""
        for doc in docs:
            text += doc.page_content
        return text
