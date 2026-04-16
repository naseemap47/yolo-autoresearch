from langchain_ollama import ChatOllama
from subagents.research_agent import research_agent
from subagents.train_agent import train_agent
from deepagents import create_deep_agent
from deepagents.backends import StateBackend


model = ChatOllama(
    model="qwen3.5:9b"
)

agent = create_deep_agent(
    model=model,
    subagents=[
        research_agent(),
        train_agent()
    ],
    backend=StateBackend(),
    system_prompt=(
        "You are the Lead Coordinator for the YOLO Auto-Research system, an autonomous framework dedicated to researching, "
        "exploring, and training YOLO object detection models. Your primary role is to understand user requests, "
        "formulate an execution plan, and delegate tasks to your specialized sub-agents:\n"
        "1. `research-agent`: Use this to search the internet, arXiv, and Wikipedia for literature, best practices, "
        "and hyperparameters related to YOLO models.\n"
        "2. `train-agent`: Use this to run an actual training job using the ultralytics framework. "
        "You must provide it with precise hyperparameters based on research.\n\n"
        "Guidelines:\n"
        "- Break down complex requests into a clear step-by-step plan.\n"
        "- If requested to solve a specific problem (e.g., 'detecting small objects'), always use the research-agent first to "
        "find optimal hyperparameters before training.\n"
        "- Synthesize all results returned by your sub-agents in a concise and clear manner.\n"
        "- Do not hallucinate values; rely on your research-agent.\n"
    ),
)


for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "train yolo model on /home/eternalcm/Projects/NHB/full_data/Train_yolo_data/data.yaml, find the suitable yolo model fit for this dataset"}]},
    stream_mode="updates",
    subgraphs=True,
    version="v2",
):
    if chunk["type"] == "updates":
        if chunk["ns"]:
            # Subagent event - namespace identifies the source
            print(f"[subagent: {chunk['ns']}]")
        else:
            # Main agent event
            print("[main agent]")
        print(chunk["data"])