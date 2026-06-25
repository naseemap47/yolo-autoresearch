import os
import yaml
import glob
from typing import Dict, Any, List

def analyze_yolo_dataset(dataset_path: str, max_scan_images: int = 200) -> Dict[str, Any]:
    """
    Analyzes a YOLO format dataset at dataset_path.
    
    Reads data.yaml to get classes and paths, then scans training labels
    to compute bounding box size distributions (small, medium, large)
    and class distributions.
    
    Bbox size criteria (relative area to image):
      - Small: area < 0.002 (~32x32 pixels on a 640x640 image)
      - Medium: 0.002 <= area <= 0.02 (~32x32 to ~90x90 pixels)
      - Large: area > 0.02 (> 90x90 pixels)
    """
    # 1. Find and parse data.yaml
    # Check if dataset_path is a directory containing data.yaml, or direct path to data.yaml
    yaml_path = None
    if os.path.isdir(dataset_path):
        possible_yamls = glob.glob(os.path.join(dataset_path, "*.yaml"))
        if possible_yamls:
            yaml_path = possible_yamls[0]
        else:
            # Check for common names
            for name in ["data.yaml", "dataset.yaml"]:
                p = os.path.join(dataset_path, name)
                if os.path.exists(p):
                    yaml_path = p
                    break
    elif os.path.isfile(dataset_path) and (dataset_path.endswith(".yaml") or dataset_path.endswith(".yml")):
        yaml_path = dataset_path
        dataset_path = os.path.dirname(dataset_path)

    if not yaml_path or not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Could not find a YAML configuration file in/at: {dataset_path}")

    with open(yaml_path, "r") as f:
        data_cfg = yaml.safe_load(f)

    # Extract class names
    classes = data_cfg.get("names", {})
    if isinstance(classes, list):
        # Convert list to dict format
        classes = {i: name for i, name in enumerate(classes)}
    elif not isinstance(classes, dict):
        classes = {}
        
    num_classes = len(classes)
    class_names = list(classes.values())

    # Determine train images directory
    train_images_sub = data_cfg.get("train", "train/images")
    base_path = data_cfg.get("path", "")
    
    # Resolve the path properly
    if base_path:
        # If path is defined, use it as prefix (could be relative to data.yaml folder or absolute)
        if not os.path.isabs(base_path):
            base_path = os.path.abspath(os.path.join(os.path.dirname(yaml_path), base_path))
        train_img_dir = os.path.join(base_path, train_images_sub)
    else:
        # Otherwise, check relative to the yaml file directory
        train_img_dir = os.path.join(os.path.dirname(yaml_path), train_images_sub)
        if not os.path.exists(train_img_dir):
            # Try direct path
            train_img_dir = os.path.join(dataset_path, train_images_sub)

    # Standardize path
    train_img_dir = os.path.abspath(train_img_dir)
    if not os.path.exists(train_img_dir):
        # If it doesn't exist, search the dataset folder for any directory ending in "images" containing files
        found = False
        for root, dirs, files in os.walk(dataset_path):
            if root.endswith("images") and any(f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')) for f in files):
                train_img_dir = root
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Train images directory not found at: {train_img_dir}")

    # Gather image files
    img_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.webp']
    all_images = []
    for ext in img_extensions:
        all_images.extend(glob.glob(os.path.join(train_img_dir, f"*{ext}")))
        all_images.extend(glob.glob(os.path.join(train_img_dir, f"*{ext.upper()}")))

    total_images = len(all_images)
    if total_images == 0:
        raise ValueError(f"No image files found in train directory: {train_img_dir}")

    # 2. Analyze label bounding boxes
    # Find matching label files
    # YOLO labels are in a sibling "labels" directory or in the same directory, replacing "/images/" with "/labels/"
    # or replacing file extension with .txt
    scanned_images = 0
    scanned_boxes = 0
    
    small_boxes = 0
    medium_boxes = 0
    large_boxes = 0
    
    class_counts = {cid: 0 for cid in classes.keys()}
    
    # Scan a subset of images to be fast, but representative
    scan_step = max(1, total_images // max_scan_images)
    images_to_scan = all_images[::scan_step][:max_scan_images]

    for img_path in images_to_scan:
        # Try to find corresponding label file
        # Check standard layout: replace "/images/" with "/labels/" and extension with ".txt"
        label_path = img_path.replace("/images/", "/labels/")
        # Also change file extension
        base_no_ext, _ = os.path.splitext(label_path)
        label_path = base_no_ext + ".txt"

        if not os.path.exists(label_path):
            # Try sibling folder search
            # E.g. image is in train/images/abc.jpg -> label in train/labels/abc.txt
            img_dir = os.path.dirname(img_path)
            parent_dir = os.path.dirname(img_dir)
            label_path = os.path.join(parent_dir, "labels", os.path.basename(base_no_ext) + ".txt")

        if not os.path.exists(label_path):
            # Try same folder
            label_path = base_no_ext + ".txt"

        if os.path.exists(label_path):
            scanned_images += 1
            try:
                with open(label_path, "r") as lf:
                    lines = lf.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        w = float(parts[3])
                        h = float(parts[4])
                        area = w * h
                        
                        scanned_boxes += 1
                        # Increment class count
                        if class_id in class_counts:
                            class_counts[class_id] += 1
                        else:
                            class_counts[class_id] = 1

                        # Bbox classification
                        if area < 0.002:
                            small_boxes += 1
                        elif area <= 0.02:
                            medium_boxes += 1
                        else:
                            large_boxes += 1
            except Exception as e:
                # Silently skip file read errors
                continue

    # Compute percentages
    small_pct = round((small_boxes / scanned_boxes) * 100, 2) if scanned_boxes > 0 else 0.0
    medium_pct = round((medium_boxes / scanned_boxes) * 100, 2) if scanned_boxes > 0 else 0.0
    large_pct = round((large_boxes / scanned_boxes) * 100, 2) if scanned_boxes > 0 else 0.0
    
    # Class distribution details
    class_distribution = {classes.get(cid, str(cid)): count for cid, count in class_counts.items()}

    return {
        "yaml_path": yaml_path,
        "train_img_dir": train_img_dir,
        "num_classes": num_classes,
        "class_names": class_names,
        "total_images": total_images,
        "scanned_images": scanned_images,
        "total_boxes": scanned_boxes,
        "bbox_distribution": {
            "small": small_pct,
            "medium": medium_pct,
            "large": large_pct
        },
        "class_distribution": class_distribution
    }
