from mcp.server.fastmcp import FastMCP
from tools import *
from dotenv import load_dotenv

# Load env
load_dotenv()

# MCP - FastAPI
mcp = FastMCP("docs")

@mcp.tool()
async def web_search(query: str):
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

@mcp.tool()
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

@mcp.tool()
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

if __name__ == "__main__":
    mcp.run(transport="stdio")
