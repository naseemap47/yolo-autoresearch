import json
import re
from typing import Optional
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from .prompt import PROMPT_WITH_CONTEXT


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
        model_name: str = "qwen3.5:9b",
        retriever: Optional[object] = None,
        target_map: float = 0.8,
    ):
        self.llm = OllamaLLM(model=model_name)
        self.retriever = retriever
        self.target_map = target_map

        if retriever is not None:
            self.prompt = PromptTemplate(
                input_variables=["context", "dataset_stats", "history", "target_map"],
                template=PROMPT_WITH_CONTEXT,
            )
            print("[Planner] RAG retriever attached — documentation context enabled.")
        else:
            raise Exception("[Planner] No RAG retriever — documentation context required.")
            # self.prompt = PromptTemplate(
            #     input_variables=["dataset_stats", "history"],
            #     template=_PROMPT_NO_CONTEXT,
            # )
            # print("[Planner] No RAG retriever — running without documentation context.")

    def plan(self, dataset_stats: dict, history: list) -> dict:
        chain = self.prompt | self.llm

        # Build prompt kwargs
        invoke_kwargs = {
            "dataset_stats": json.dumps(dataset_stats, indent=2),
            "history": json.dumps(history, indent=2),
            "target_map": self.target_map,
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
            print(f"[Planner] Failed to parse LLM response: {result}")
            print("[INFO] Might be due to LLM memory issue!\nUpgrade to higher model")
            raise Exception("[Planner] Failed to parse LLM response.")
            # return {
            #     "model_name": "yolov8n.pt",
            #     "epochs": 10,
            #     "batch_size": 16,
            #     "imgsz": 640,
            #     "lr0": 0.01,
            #     "weight_decay": 0.0005,
            #     "close_mosaic": 10,
            #     "reasoning": "Fallback configuration due to JSON parsing error.",
            # }

