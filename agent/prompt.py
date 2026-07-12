# Prompt used when a RAG retriever is available (with documentation context)
PROMPT_WITH_CONTEXT = """\
# YOLO Object Detection - Automated Model & Hyperparameter Optimization Plan

Below is the documentation, dataset statistics, and history of previous training runs.
A research system analyzes this information and outputs the optimal model configuration in JSON format.

---

## 1. Ultralytics Documentation Context
{context}

---

## 2. Parameter Reference Guide

### Available Base Models (choose the model file name, e.g. "yolo11n.pt"):
- Nano/Tiny/Small: yolo26n.pt, yolo26s.pt, yolo12n.pt, yolo12s.pt, yolo11n.pt, yolo11s.pt, yolov8n.pt, yolov8s.pt, yolov5nu.pt, yolov5su.pt, yolov10n.pt, yolov10s.pt, yolov9t.pt, yolov9s.pt
- Medium/Compact: yolo26m.pt, yolo12m.pt, yolo11m.pt, yolov8m.pt, yolov5mu.pt, yolov10m.pt, yolov9m.pt, yolov9c.pt
- Large/XLarge: yolo26l.pt, yolo26x.pt, yolo12l.pt, yolo12x.pt, yolo11l.pt, yolo11x.pt, yolov8l.pt, yolov8x.pt, yolov5lu.pt, yolov5xu.pt, yolov10l.pt, yolov10x.pt, yolov9e.pt, rtdetr-l.pt, rtdetr-x.pt

### Key Training Hyperparameters:
- epochs: Total training epochs (e.g. 50).
- batch_size: Batch size (e.g., 16, -1 for auto, 0.70 for 70% GPU memory limit).
- imgsz: Input image size (e.g., 640). Preserves aspect ratio.
- optimizer: SGD, MuSGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, or auto.
- lr0: Initial learning rate (e.g. 0.01).
- weight_decay: L2 regularization term (e.g. 0.0005).
- close_mosaic: Disables mosaic augmentation in final N epochs (e.g. 10). Set to 0 to disable.
- single_cls: True to treat all classes as one class.
- freeze: Number of layers to freeze (e.g., 10 to freeze backbone) or list of layer indices.
- cos_lr: True to use cosine learning rate scheduler.
- amp: True to enable Automatic Mixed Precision.
- patience: Early stopping patience epochs (e.g. 20).
- distill_model: Path to a teacher model checkpoint (e.g. "yolov8l.pt") for knowledge distillation.

### Key Augmentation Settings:
- hsv_h, hsv_s, hsv_v: HSV color space augmentations.
- degrees: Rotation range in degrees.
- translate: Translation range fraction.
- scale: Scale gain range.
- shear: Shear range in degrees.
- perspective: Perspective range.
- flipud: Vertical flip probability.
- fliplr: Horizontal flip probability.
- mosaic: Mosaic augmentation probability.
- mixup: Mixup augmentation probability.
- cutmix: CutMix augmentation probability.

---

## 3. Training Optimization Reference Examples

### Example 1: Small Dataset with Edge Device Constraint
- **Target mAP**: 0.8
- **Dataset Statistics**:
{{
  "num_classes": 1,
  "class_names": ["defect"],
  "total_images": 150,
  "processed_images": 150,
  "empty_images": 5,
  "image_size_stats": {{
    "average_width": 512.0,
    "average_height": 512.0,
    "min_width": 512,
    "max_width": 512,
    "min_height": 512,
    "max_height": 512,
    "average_aspect_ratio": 1.0
  }},
  "bbox_stats": {{
    "total_bboxes": 180,
    "average_bboxes_per_image": 1.2,
    "min_bboxes_per_image": 1,
    "max_bboxes_per_image": 3,
    "average_relative_width": 0.04,
    "average_relative_height": 0.04,
    "min_relative_width": 0.01,
    "max_relative_width": 0.08,
    "min_relative_height": 0.01,
    "max_relative_height": 0.08,
    "average_absolute_width": 20.48,
    "average_absolute_height": 20.48,
    "min_absolute_width": 5.12,
    "max_absolute_width": 40.96,
    "min_absolute_height": 5.12,
    "max_absolute_height": 40.96,
    "average_aspect_ratio": 1.0
  }},
  "class_distribution": {{
    "defect": {{
      "class_id": 0,
      "count": 180,
      "percentage": 100.0,
      "avg_relative_width": 0.04,
      "avg_relative_height": 0.04,
      "avg_absolute_width": 20.48,
      "avg_absolute_height": 20.48
    }}
  }}
}}
- **Training History**:
[]
- **Recommended Configuration**:
```json
{{
  "model_name": "yolo11n.pt",
  "epochs": 50,
  "batch_size": 16,
  "imgsz": 512,
  "optimizer": "AdamW",
  "lr0": 0.001,
  "lrf": 0.01,
  "momentum": 0.9,
  "weight_decay": 0.01,
  "warmup_epochs": 3.0,
  "warmup_momentum": 0.8,
  "warmup_bias_lr": 0.1,
  "distill_model": "",
  "cos_lr": false,
  "close_mosaic": 15,
  "amp": true,
  "patience": 20,
  "single_cls": true,
  "multi_scale": 0.0,
  "freeze": 0,
  "dropout": 0.1,
  "box": 7.5,
  "cls": 0.5,
  "dfl": 1.5,
  "hsv_h": 0.015,
  "hsv_s": 0.7,
  "hsv_v": 0.4,
  "degrees": 0.0,
  "translate": 0.1,
  "scale": 0.5,
  "shear": 0.0,
  "perspective": 0.0,
  "flipud": 0.0,
  "fliplr": 0.5,
  "bgr": 0.0,
  "mosaic": 0.5,
  "mixup": 0.0,
  "cutmix": 0.0,
  "reasoning": "Small dataset (150 images) with small objects (~20px). Switched to yolo11n for lightweight fast inference, used AdamW and weight_decay of 0.01 to prevent overfitting, lowered mosaic to 0.5 to reduce context noise, and matched imgsz to dataset average of 512."
}}
```

---

## 4. Current Optimization Task
- **Target mAP**: {target_map}
- **Dataset Statistics**:
{dataset_stats}
- **Training History**:
{history}
- **Recommended Configuration**:
"""
