from langchain_ollama import ChatOllama
from tools.seach_tool import internet_search, wiki_search, arxiv_search

subagent_model = ChatOllama(
    model="qwen3.5:9b"
)

def research_agent():
    return {
        "name": "research-agent",
        "description": "Used to research more in depth questions",
        "system_prompt": (
            "You are the Lead AI Researcher for the YOLO Auto-Research system. Your expertise focuses on computer vision "
            "and the YOLO family of object detection models. Your primary objective is to find high-quality information, "
            "academic papers, and tutorials regarding YOLO training, modifications, and hyperparameter tuning.\n\n"
            "Guidelines:\n"
            "- Use your search tools (internet, wiki, arxiv) exhaustively to gather factual, up-to-date data.\n"
            "- Always cite your sources when summarizing findings.\n"
            "- If asked for hyperparameter recommendations, search for empirical optimal values for the specific task.\n"
            "- Return your findings in a structured, detailed, and easy-to-read format."
        ),
        "tools": [internet_search, wiki_search, arxiv_search],
        "model": subagent_model,
    }
