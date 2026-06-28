import json
import re
from typing import Optional
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# Prompt used when a RAG retriever is available (with documentation context)
_PROMPT_WITH_CONTEXT = """\
You are an AI auto-researcher specializing in YOLO object detection models.

## Relevant Ultralytics Documentation
The following excerpts from official Ultralytics documentation are provided to \
help you make evidence-based decisions about hyperparameters and model selection:

{context}

---

Given the dataset statistics and history of previous training runs (if any), \
propose the next YOLO model architecture and hyperparameters. \
Use the documentation excerpts above to justify your choices.

Dataset Statistics:
{dataset_stats}

Training History:
{history}

Your goal is to reach a mAP50-95 > 0.8.
Available Base Models: yolov8n.pt, yolov8s.pt, yolov8m.pt
Hyperparameters to tune:
  - epochs      : 10 – 50
  - batch_size  : 8, 16, or 32
  - imgsz       : 320 or 640
  - lr0         : 0.001 – 0.01
  - weight_decay: 0.0001 – 0.001  (optional, helps with overfitting)
  - close_mosaic: 0 – 15          (optional, epochs before end to disable mosaic)

Respond ONLY with a valid JSON object in the following format:
{{
  "model_name": "yolov8n.pt",
  "epochs": 10,
  "batch_size": 16,
  "imgsz": 640,
  "lr0": 0.01,
  "weight_decay": 0.0005,
  "close_mosaic": 10,
  "reasoning": "Brief explanation referencing the docs above."
}}

JSON Response:"""

# Prompt used when no retriever is available (backwards-compatible fallback)
_PROMPT_NO_CONTEXT = """\
You are an AI auto-researcher specializing in YOLO object detection models.

Given the dataset statistics and the history of previous model trainings (if any), \
propose the next YOLO model architecture and hyperparameters.

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


def _build_rag_query(dataset_stats: dict, history: list) -> str:
    """
    Build a natural-language query that captures the planning problem so the
    RAG retriever can surface the most relevant documentation chunks.
    """
    parts = []

    num_classes = dataset_stats.get("num_classes", 0)
    total_images = dataset_stats.get("total_images", 0)
    parts.append(
        f"YOLO training with {num_classes} classes and {total_images} images. "
        "How should I choose learning rate, batch size, epochs, weight decay, "
        "and data augmentation settings?"
    )

    if history:
        last = history[-1]
        map_val = last.get("results", {}).get("mAP50-95", None)
        if map_val is not None:
            if map_val < 0.3:
                parts.append(
                    "The previous run achieved a very low mAP. "
                    "What hyperparameters help avoid underfitting or overfitting on small datasets?"
                )
            elif map_val < 0.6:
                parts.append(
                    "The previous run achieved moderate mAP. "
                    "How can I improve mAP with learning rate scheduling, "
                    "regularization, or augmentation?"
                )
            else:
                parts.append(
                    "The previous run achieved good mAP. "
                    "How should I fine-tune hyperparameters to push mAP higher?"
                )

    if total_images < 500:
        parts.append(
            "Small dataset overfitting prevention: dropout, weight decay, "
            "data augmentation best practices."
        )

    return " ".join(parts)


class Planner:
    def __init__(
        self,
        model_name: str = "qwen3.5:0.8b",
        retriever: Optional[object] = None,
    ):
        self.llm = OllamaLLM(model=model_name)
        self.retriever = retriever

        if retriever is not None:
            self.prompt = PromptTemplate(
                input_variables=["context", "dataset_stats", "history"],
                template=_PROMPT_WITH_CONTEXT,
            )
            print("[Planner] RAG retriever attached — documentation context enabled.")
        else:
            self.prompt = PromptTemplate(
                input_variables=["dataset_stats", "history"],
                template=_PROMPT_NO_CONTEXT,
            )
            print("[Planner] No RAG retriever — running without documentation context.")

    def plan(self, dataset_stats: dict, history: list) -> dict:
        chain = self.prompt | self.llm

        # Build prompt kwargs
        invoke_kwargs = {
            "dataset_stats": json.dumps(dataset_stats, indent=2),
            "history": json.dumps(history, indent=2),
        }

        if self.retriever is not None:
            query = _build_rag_query(dataset_stats, history)
            context = self.retriever.query(query)
            invoke_kwargs["context"] = context
            print(f"[Planner] RAG query: {query[:120]}…")
            print(f"[Planner] Retrieved {len(context)} chars of documentation context.")

        result = chain.invoke(invoke_kwargs)

        # Parse JSON from LLM response
        try:
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                result = json_match.group(0)

            config = json.loads(result)
            return config
        except Exception:
            print(f"Failed to parse LLM response: {result}")
            return {
                "model_name": "yolov8n.pt",
                "epochs": 10,
                "batch_size": 16,
                "imgsz": 640,
                "lr0": 0.01,
                "weight_decay": 0.0005,
                "close_mosaic": 10,
                "reasoning": "Fallback configuration due to JSON parsing error.",
            }

