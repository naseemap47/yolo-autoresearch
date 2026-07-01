# Prompt used when a RAG retriever is available (with documentation context)
PROMPT_WITH_CONTEXT = """\
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

Train Settings:
  - epochs          : (default: 50) Total number of training epochs. Each epoch represents a full pass over the entire dataset. Adjusting this value can affect training duration and model performance.
  - batch_size      : (default: 16) Batch size, with three modes: set as an integer (e.g., batch=16), auto mode for 60% GPU memory utilization (batch=-1), or auto mode with specified utilization fraction (batch=0.70).
  - imgsz           : (default: 640) Target image size for training. Images are resized to squares with sides equal to the specified value (if rect=False), preserving aspect ratio for YOLO models but not RT-DETR. Affects model accuracy and computational complexity.
  - lr0             : (default: 0.01) Initial learning rate (i.e. SGD=1E-2, Adam=1E-3). Adjusting this value is crucial for the optimization process, influencing how rapidly model weights are updated. It's recommended to start with 0.01 and decrease by 0.001 if needed. 
  - weight_decay    : (default: 0.0005) Weight decay is a regularization technique that helps to prevent overfitting. It's recommended to use a weight decay of 0.0005 to 0.001. Higher values will prevent overfitting but may also reduce model performance. L2 regularization term, penalizing large weights to prevent overfitting.
  - close_mosaic    : (optional) Disables mosaic data augmentation in the last N epochs to stabilize training before completion. Setting to 0 disables this feature. if you want add this feature , set value from 5 to 20. Increase close_mosaic value when the model is overfitting.
  - patience        : (default: 20) Number of epochs to wait without improvement in validation metrics before early stopping the training. Helps prevent overfitting by stopping training when performance plateaus.
  - optimizer       : Choice of optimizer for training. Options include SGD, MuSGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, or auto for automatic selection based on model configuration. Affects convergence speed and stability.
  - single_cls      : (default: False) Treats all classes in multi-class datasets as a single class during training. Useful for binary classification tasks or when focusing on object presence rather than classification.
  - multi_scale     : (default: 0.0) Randomly vary imgsz each batch by +/- multi_scale (e.g. 0.25 -> 0.75x to 1.25x), rounding to model stride multiples; 0.0 disables multi-scale training.
  - cos_lr          : (default: False) Utilizes a cosine learning rate scheduler, adjusting the learning rate following a cosine curve over epochs. Helps in managing learning rate for better convergence.
  - close_mosaic    : (default: 10) Disables mosaic data augmentation in the last N epochs to stabilize training before completion. Setting to 0 disables this feature.
  - amp             : (default: True) Enables Automatic Mixed Precision (AMP) training, reducing memory usage and possibly speeding up training with minimal impact on accuracy.
  - freeze          : (default: 0) The freeze parameter accepts either an integer or a list. An integer freeze=10 freezes the first 10 layers (indices 0-9), which covers most of the YOLO26 backbone. The backbone spans layers 0-10, so freeze=10 leaves the final C2PSA block (layer 10) trainable; use freeze=11 to freeze the entire backbone. A list can contain layer indices like freeze=[0, 3, 5] for partial backbone freezing, or module name strings like freeze=["23.cv2", "23.one2one_cv2"] for fine-grained control over specific branches within a layer (here, both box regression branches of the detection head).
  - lrf             : (default: 0.01) Final learning rate as a fraction of the initial rate = (lr0 * lrf), used in conjunction with schedulers to adjust the learning rate over time.
  - momentum        : (default: 0.937) Momentum factor for SGD or beta1 for Adam optimizers, influencing the incorporation of past gradients in the current update.
  - warmup_epochs   : (default: 3.0) Number of epochs for learning rate warmup, gradually increasing the learning rate from a low value to the initial learning rate to stabilize training early on.
  - warmup_momentum : (default: 0.8) Initial momentum for warmup phase, gradually adjusting to the set momentum over the warmup period.
  - warmup_bias_lr  : (default: 0.1) Learning rate for bias parameters during the warmup phase, helping stabilize model training in the initial epochs.
  - distill_model   : Path to a teacher model checkpoint (e.g. yolo26x.pt) for knowledge distillation. When set, the student model is trained with an extra distillation loss guided by the frozen teacher.
  - dis             : (default: 6.0) Weight of the distillation loss added to the standard detection losses. Higher values increase the influence of the teacher's feature guidance.
  - box             : (default: 7.5) Weight of the box loss component in the loss function, influencing how much emphasis is placed on accurately predicting bounding box coordinates.
  - cls             : (default: 0.5) Weight of the classification loss in the total loss function, affecting the importance of correct class prediction relative to other components.
  - cls_pw          : (default: 0.0) Power for class weighting to handle class imbalance using inverse class frequency. 0.0 disables class weighting, 1.0 applies full inverse frequency weighting. Values between 0 and 1 provide partial weighting.
  - dfl             : (default: 1.5) Weight of the distribution focal loss, used in certain YOLO versions for fine-grained classification.
  - nbs             : (default: 64) Nominal batch size for normalization of loss.
  - overlap_mask    : (default: True) Determines whether object masks should be merged into a single mask for training, or kept separate for each object. In case of overlap, the smaller mask is overlaid on top of the larger mask during merge.
  - mask_ratio      : (default: 4) Downsample ratio for segmentation masks, affecting the resolution of masks used during training.
  - dropout         : (default: 0.0) Dropout rate for regularization in classification tasks, preventing overfitting by randomly omitting units during training.
  - compile         : (default: False) Enables PyTorch 2.x torch.compile graph compilation with backend='inductor'. Accepts True → "default", False → disables, or a string mode such as "default", "reduce-overhead", "max-autotune-no-cudagraphs". Falls back to eager with a warning if unsupported.
  - max_det         : (default: 300) Specifies the maximum number of objects retained during validation phase of training.

Augmentation Settings and Hyperparameters:
  - hsv_h           : (default: 0.015) HSV Hue augmentation range (0.0 - 1.0). Adjusts the hue of the image by a fraction of the color wheel, introducing color variability. Helps the model generalize across different lighting conditions.
  - hsv_s           : (default: 0.7) HSV Saturation augmentation range (0.0 - 0.1). Alters the saturation of the image by a fraction, affecting the intensity of colors. Useful for simulating different environmental conditions.
  - hsv_v           : (default: 0.4) HSV Value augmentation range (0.0 - 0.1). Modifies the value (brightness) of the image by a fraction, helping the model to perform well under various lighting conditions.
  - degrees         : (default: 0.0) Rotation range in degrees (0.0 - 180). Rotates the image randomly within the specified degree range, improving the model's ability to recognize objects at various orientations.
  - translate       : (default: 0.1) Translation range (0.0 - 0.1). Translates the image horizontally and vertically by a fraction of the image size, aiding in learning to detect partially visible objects.
  - scale           : (default: 0.5) Scaling range (0.0 - 1.0). Scales the image by a gain factor, simulating objects at different distances from the camera.
  - shear           : (default: 0.0) Shear augmentation range (-180 to 180). Shears the image by a specified degree, mimicking the effect of objects being viewed from different angles.
  - perspective     : (default: 0.0) Perspective transformation range (0.0 - 0.001). Applies a random perspective transformation to the image, enhancing the model's ability to understand objects in 3D space.
  - flipud          : (default: 0.0) Vertical flip probability (0.0 - 1.0). Flips the image upside down with the specified probability, increasing the data variability without affecting the object's characteristics.
  - fliplr          : (default: 0.5) Horizontal flip probability (0.0 - 1.0). Flips the image left to right with the specified probability, useful for learning symmetrical objects and increasing dataset diversity.
  - bgr             : (default: 0.0) range 0.0 - 1.0. Flips the image channels from RGB to BGR with the specified probability, useful for increasing robustness to incorrect channel ordering.
  - mosaic          : (default: 1.0) Mosaic augmentation probability (0.0 - 1.0). Combines four training images into one, simulating different scene compositions and object interactions. Highly effective for complex scene understanding.
  - mixup           : (default: 0.0) Mixup augmentation probability (0.0 - 1.0). Blends two images and their labels, creating a composite image. Enhances the model's ability to generalize by introducing label noise and visual variability.
  - cutmix          : (default: 0.0) CutMix augmentation probability (0.0 - 1.0). Combines portions of two images, creating a partial blend while maintaining distinct regions. Enhances model robustness by creating occlusion scenarios.

Note on Batch-size Settings:
The batch argument can be configured in three ways:

Fixed Batch Size: Set an integer value (e.g., batch=16), specifying the number of images per batch directly.
Auto Mode (60% GPU Memory): Use batch=-1 to automatically adjust batch size for approximately 60% CUDA memory utilization.
Auto Mode with Utilization Fraction: Set a fraction value (e.g., batch=0.70) to adjust batch size based on the specified fraction of GPU memory usage.
OOM Auto-Retry: If a CUDA out-of-memory error occurs during the first epoch, the trainer automatically halves the batch size and retries (up to 3 times). This only applies to single-GPU training; multi-GPU (DDP) training will raise the error immediately.

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

