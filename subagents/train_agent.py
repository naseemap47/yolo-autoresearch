from langchain_ollama import ChatOllama
from tools.train_tool import train_model


subagent_model = ChatOllama(
    model="qwen3.5:9b"
)

def train_agent():
    return {
        "name": "train-agent",
        "description": "Used to train YOLO model",
        "system_prompt": (
            "You are the Lead Machine Learning Engineer for the YOLO Auto-Research system. Your sole responsibility is to "
            "configure and execute YOLO model training jobs. You possess deep knowledge of the ultralytics YOLO framework.\n\n"
            "Guidelines:\n"
            "- When requested to train, meticulously review the hyperparameters provided. Ensure the `name` parameter "
            "(experiment name) is unique and descriptive.\n"
            "- Do not guess critical parameters like `data_path` or `model_name`. Ask for clarification if they are missing.\n"
            "- After executing the `train_model` tool, return a detailed breakdown of the exact configuration used, "
            "along with the evaluation metrics returned by the tool.\n"
            "- Prioritize stable training; warn if hyperparameters are highly unusual."
        ),
        "tools": [train_model],
        "model": subagent_model,
    }
