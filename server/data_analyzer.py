import os
import yaml
import glob
from collections import defaultdict
from PIL import Image

def analyze_yolo_dataset(data_yaml_path: str):
    if not os.path.exists(data_yaml_path):
        return {"error": "Dataset yaml not found."}
        
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        
    # Get paths relative to yaml
    base_dir = os.path.dirname(data_yaml_path)
    train_path = os.path.join(base_dir, data.get('train', ''))
    
    if not os.path.exists(train_path):
        # try as relative to current dir? Usually it's relative to yaml or absolute
        if os.path.isabs(data.get('train', '')):
            train_path = data.get('train', '')
        else:
            return {"error": f"Train path not found: {train_path}"}
            
    # Find all images
    # YOLO typically uses images/ and labels/
    # If train_path is a txt file (list of images)
    image_paths = []
    if os.path.isfile(train_path) and train_path.endswith('.txt'):
        with open(train_path, 'r') as f:
            image_paths = [line.strip() for line in f.readlines()]
    elif os.path.isdir(train_path):
        image_paths = glob.glob(os.path.join(train_path, '**', '*.*'), recursive=True)
        # Filter for images
        image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]

    stats = {
        "num_classes": len(data.get('names', [])),
        "class_names": data.get('names', []),
        "total_images": len(image_paths),
        "average_image_width": 0,
        "average_image_height": 0,
        "bbox_distribution": defaultdict(list),
        "class_counts": defaultdict(int)
    }
    
    total_w, total_h = 0, 0
    sampled_count = 0
    
    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                total_w += w
                total_h += h
                
            # Find label
            # Typical YOLO format: replace /images/ with /labels/ and extension with .txt
            label_path = img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
            if os.path.exists(label_path):
                with open(label_path, 'r') as lf:
                    lines = lf.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            bx, by, bw, bh = map(float, parts[1:5])
                            stats["class_counts"][cls_id] += 1
                            # store relative bbox area
                            stats["bbox_distribution"][cls_id].append(bw * bh)
                            
            sampled_count += 1
        except Exception as e:
            continue
            
    stats["sampled_images"] = sampled_count
    if sampled_count > 0:
        stats["average_image_width"] = total_w / sampled_count
        stats["average_image_height"] = total_h / sampled_count
        
    # Summarize bbox
    bbox_summary = {}
    for cls_id, areas in stats["bbox_distribution"].items():
        if areas:
            bbox_summary[cls_id] = {
                "avg_relative_area": sum(areas) / len(areas),
                "count": len(areas)
            }
            
    stats["bbox_summary"] = bbox_summary
    del stats["bbox_distribution"]
    
    return stats
