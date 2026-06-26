import json
import re
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

class Planner:
    def __init__(self, model_name: str = "qwen3.5:0.8b"):
        self.llm = OllamaLLM(model=model_name)
        self.prompt = PromptTemplate(
            input_variables=["dataset_stats", "history"],
            template="""You are an AI auto-researcher specializing in YOLO object detection models.

Given the dataset statistics and the history of previous model trainings (if any), propose the next YOLO model architecture and hyperparameters.

Dataset Statistics:
{dataset_stats}

Training History:
{history}

Your goal is to reach a mAP50-95 > 0.8.
Available Base Models: yolov8n.pt, yolov8s.pt, yolov8m.pt
Hyperparameters to tune: epochs (max 50), batch_size (8, 16, 32), imgsz (320, 640), lr0 (0.001 to 0.01)

Respond ONLY with a valid JSON object in the following format:
{{
  "model_name": "yolov8n.pt",
  "epochs": 10,
  "batch_size": 16,
  "imgsz": 640,
  "lr0": 0.01,
  "reasoning": "Brief explanation of why you chose these parameters."
}}

JSON Response:"""
        )
        
    def plan(self, dataset_stats: dict, history: list):
        chain = self.prompt | self.llm
        result = chain.invoke({
            "dataset_stats": json.dumps(dataset_stats, indent=2),
            "history": json.dumps(history, indent=2)
        })
        
        # Parse JSON
        try:
            # simple cleanup if LLM returns markdown fences
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()
                
            # Regex to find the JSON block even if there is surrounding text
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                result = json_match.group(0)
                
            config = json.loads(result)
            return config
        except Exception as e:
            print(f"Failed to parse LLM response: {result}")
            # Fallback configuration
            return {
                "model_name": "yolov8n.pt",
                "epochs": 10,
                "batch_size": 16,
                "imgsz": 640,
                "lr0": 0.01,
                "reasoning": "Fallback configuration due to JSON parsing error."
            }
