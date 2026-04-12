from mcp.server.fastmcp import FastMCP
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.retrievers import ArxivRetriever
from dotenv import load_dotenv

# Load env
load_dotenv()

# MCP - FastAPI
mcp = FastMCP("docs")

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

@mcp.tool()
async def web_search(query: str):
    """
    Web Search Tool to search and read docs, articles and News from web.
    For Wikipedia search use wiki_search tool.
    
    Args:
        query: The query to search for (eg. "Who is Elon Musk ?", "How to integrate chroma db with langchain ?")
    
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

@mcp.tool()
async def wiki_search(query: str):
    """
    Wikipedia search tool to search and and get the details about the query

    Args:
        query: The query to search for (eg. "Who is Elon Musk ?", "Japan history")
    
    Returns:
        extracted text from wikipedia about the query
    """
    retriever = WikipediaRetriever()
    docs = await retriever.invoke(query)
    if len(docs) == 0:
        return "No results found"
    else:
        text = ""
        for doc in docs:
            text += doc.page_content
        return text

@mcp.tool()
async def arxiv_search(query: str):
    """
    ArXiv search tool to search about research papers from ArXiv.

    Args:
        query: The query to search for (eg. "Expain me about 1605.08386 research paper.", "give me details about Transformers from Ai model architecture")
    
    Returns:
        extracted text from ArXiv about the query
    """
    retriever = ArxivRetriever(
        load_max_docs=2,
        get_full_documents=True,
    )
    docs = await retriever.invoke(query)
    if len(docs) == 0:
        return "No results found"
    else:
        text = ""
        for doc in docs:
            text += doc.page_content
        return text

if __name__ == "__main__":
    mcp.run(transport="stdio")
