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

Your goal is to reach a mAP50-95 > {target_map}.
Available Base Models (choose the model file name, e.g. "yolo11n.pt"):
| Model File       | Architecture | Size   | Characteristic/Feature |
|:-----------------|:-------------|:-------|:---|
| yolo26n.pt       | YOLO26       | Nano   | Lightest edge variant with NMS-free inference |
| yolo26s.pt       | YOLO26       | Small  | Compact edge variant with NMS-free inference |
| yolo26m.pt       | YOLO26       | Medium | Balanced edge variant with NMS-free inference |
| yolo26l.pt       | YOLO26       | Large  | Accurate edge variant with NMS-free inference |
| yolo26x.pt       | YOLO26       | XLarge | Highest-accuracy edge variant with NMS-free inference |
| yolo12n.pt       | YOLO12       | Nano   | Lightest attention-centric model with flash attention |
| yolo12s.pt       | YOLO12       | Small  | Small attention-centric model with flash attention |
| yolo12m.pt       | YOLO12       | Medium | Balanced attention-centric model with flash attention |
| yolo12l.pt       | YOLO12       | Large  | Accurate attention-centric model with flash attention |
| yolo12x.pt       | YOLO12       | XLarge | Highest-accuracy attention-centric model |
| yolo11n.pt       | YOLO11       | Nano   | Fastest multi-task versatile model |
| yolo11s.pt       | YOLO11       | Small  | Small multi-task versatile model |
| yolo11m.pt       | YOLO11       | Medium | Balanced multi-task versatile model |
| yolo11l.pt       | YOLO11       | Large  | Accurate multi-task versatile model |
| yolo11x.pt       | YOLO11       | XLarge | Highest-accuracy multi-task versatile model |
| yolov8n.pt       | YOLOv8       | Nano   | Fastest landmark standard model |
| yolov8s.pt       | YOLOv8       | Small  | Small landmark standard model |
| yolov8m.pt       | YOLOv8       | Medium | Balanced landmark standard model |
| yolov8l.pt       | YOLOv8       | Large  | Accurate landmark standard model |
| yolov8x.pt       | YOLOv8       | XLarge | Highest-accuracy landmark standard model |
| yolov5nu.pt      | YOLOv5       | Nano   | Lightest production-ready PyTorch model |
| yolov5su.pt      | YOLOv5       | Small  | Small production-ready PyTorch model |
| yolov5mu.pt      | YOLOv5       | Medium | Balanced production-ready PyTorch model |
| yolov5lu.pt      | YOLOv5       | Large  | Accurate production-ready PyTorch model |
| yolov5xu.pt      | YOLOv5       | XLarge | Highest-accuracy production-ready PyTorch model |
| yolov10n.pt      | YOLOv10      | Nano   | Fastest NMS-free ultra-low latency model |
| yolov10s.pt      | YOLOv10      | Small  | Small NMS-free ultra-low latency model |
| yolov10m.pt      | YOLOv10      | Medium | Balanced NMS-free ultra-low latency model |
| yolov10l.pt      | YOLOv10      | Large  | Accurate NMS-free ultra-low latency model |
| yolov10x.pt      | YOLOv10      | XLarge | Highest-accuracy NMS-free ultra-low latency model |
| yolov9t.pt       | YOLOv9       | Tiny   | Lightest PGI model for constrained devices |
| yolov9s.pt       | YOLOv9       | Small  | Small PGI model |
| yolov9m.pt       | YOLOv9       | Medium | Balanced PGI model |
| yolov9c.pt       | YOLOv9       | Compact| Compact PGI model optimized for speed |
| yolov9e.pt       | YOLOv9       | Extended| Highest-accuracy PGI model |
| yolov7-tiny.pt   | YOLOv7       | Tiny   | Lightest high-efficiency model for inference |
| yolov7.pt        | YOLOv7       | Medium | Standard high-efficiency model |
| yolov7x.pt       | YOLOv7       | XLarge | Highest-accuracy high-efficiency model |
| yolov6n.pt       | YOLOv6       | Nano   | Lightest industrial-use model |
| yolov6s.pt       | YOLOv6       | Small  | Small industrial-use model |
| yolov6m.pt       | YOLOv6       | Medium | Balanced industrial-use model |
| yolov6l.pt       | YOLOv6       | Large  | Accurate industrial-use model |
| yolov3u.pt       | YOLOv3       | Standard | Classic real-time detector |
| yolov3-tinyu.pt  | YOLOv3       | Tiny   | Lightweight classic real-time detector |
| yolov4-tiny.pt   | YOLOv4       | Tiny   | Lightweight Darknet-native architecture |
| yolov4.pt        | YOLOv4       | Standard | Standard Darknet-native architecture |
| yolo-worldv2s.pt | YOLO-World   | Small  | Compact open-vocabulary detector |
| yolo-worldv2m.pt | YOLO-World   | Medium | Balanced open-vocabulary detector |
| yolo-worldv2l.pt | YOLO-World   | Large  | Accurate open-vocabulary detector |
| yolo-worldv2x.pt | YOLO-World   | XLarge | Highest-accuracy open-vocabulary detector |
| yoloe-s.pt       | YOLOE        | Small  | Compact zero-shot text-prompting detector |
| yoloe-m.pt       | YOLOE        | Medium | Balanced zero-shot text-prompting detector |
| yoloe-l.pt       | YOLOE        | Large  | Accurate zero-shot text-prompting detector |
| rtdetr-l.pt      | RT-DETR      | Large  | Large Detection Transformer without NMS |
| rtdetr-x.pt      | RT-DETR      | XLarge | Highest-accuracy Detection Transformer without NMS |

Size selection guide:
- **Nano/Tiny/Small**: Ideal for real-time inference on edge/CPU devices or small datasets (<500 images).
- **Medium/Compact**: Good balance of speed and accuracy for moderate datasets.
- **Large/Extended/XLarge**: Best accuracy for large datasets or when inference speed is not the primary concern.

Hyperparameters to tune:
  - epochs      : start with 25 and increase by 25 if needed
  - batch_size  : start with 16 and increase by 16 if needed
  - imgsz       : based on image size and bbox size, you can start with 640 if you don't have specific knowledge of the dataset.
  - lr0         : Initial learning rate (i.e. SGD=1E-2, Adam=1E-3). Adjusting this value is crucial for the optimization process, influencing how rapidly model weights are updated. It's recommended to start with 0.01 and decrease by 0.001 if needed. 
  - weight_decay: Weight decay is a regularization technique that helps to prevent overfitting. It's recommended to use a weight decay of 0.0005 to 0.001. Higher values will prevent overfitting but may also reduce model performance. 
  - close_mosaic: (optional) Disables mosaic data augmentation in the last N epochs to stabilize training before completion. Setting to 0 disables this feature. if you want add this feature , set value from 5 to 20. Increase close_mosaic value when the model is overfitting.

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

# # Prompt used when no retriever is available (backwards-compatible fallback)
# _PROMPT_NO_CONTEXT = """\
# You are an AI auto-researcher specializing in YOLO object detection models.

# Given the dataset statistics and the history of previous model trainings (if any), \
# propose the next YOLO model architecture and hyperparameters.

# Dataset Statistics:
# {dataset_stats}

# Training History:
# {history}

# Your goal is to reach a mAP50-95 > 0.8.
# Available Base Models: yolov8n.pt, yolov8s.pt, yolov8m.pt
# Hyperparameters to tune: epochs (max 50), batch_size (8, 16, 32), imgsz (320, 640), lr0 (0.001 to 0.01)

# Respond ONLY with a valid JSON object in the following format:
# {{
#   "model_name": "yolov8n.pt",
#   "epochs": 10,
#   "batch_size": 16,
#   "imgsz": 640,
#   "lr0": 0.01,
#   "reasoning": "Brief explanation of why you chose these parameters."
# }}

# JSON Response:"""


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
                template=_PROMPT_WITH_CONTEXT,
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

