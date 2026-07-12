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
    image_paths = []
    if os.path.isfile(train_path) and train_path.endswith('.txt'):
        with open(train_path, 'r') as f:
            image_paths = [line.strip() for line in f.readlines()]
    elif os.path.isdir(train_path):
        image_paths = glob.glob(os.path.join(train_path, '**', '*.*'), recursive=True)
        # Filter for images
        image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]

    names_map = data.get('names', {})
    if isinstance(names_map, list):
        names_map = {i: name for i, name in enumerate(names_map)}
    elif not isinstance(names_map, dict):
        names_map = {}

    img_widths = []
    img_heights = []
    img_aspect_ratios = []
    bboxes_per_image = []
    empty_images = 0
    
    total_bboxes = 0
    bbox_rel_widths = []
    bbox_rel_heights = []
    bbox_abs_widths = []
    bbox_abs_heights = []
    bbox_aspect_ratios = []
    
    class_counts = defaultdict(int)
    class_bbox_rel_widths = defaultdict(list)
    class_bbox_rel_heights = defaultdict(list)
    class_bbox_abs_widths = defaultdict(list)
    class_bbox_abs_heights = defaultdict(list)
    
    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
            
            img_widths.append(w)
            img_heights.append(h)
            img_aspect_ratios.append(w / h if h > 0 else 0.0)
            
            # Find label
            # Typical YOLO format: replace /images/ with /labels/ and extension with .txt
            label_path = img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
            if not os.path.exists(label_path):
                # Fallback if labels are in the same directory but with a .txt extension
                alt_label_path = img_path.rsplit('.', 1)[0] + '.txt'
                if os.path.exists(alt_label_path):
                    label_path = alt_label_path
                    
            img_bboxes_count = 0
            if os.path.exists(label_path):
                with open(label_path, 'r') as lf:
                    lines = lf.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                cls_id = int(parts[0])
                                bx, by, bw, bh = map(float, parts[1:5])
                                
                                bw_px = bw * w
                                bh_px = bh * h
                                
                                bbox_rel_widths.append(bw)
                                bbox_rel_heights.append(bh)
                                bbox_abs_widths.append(bw_px)
                                bbox_abs_heights.append(bh_px)
                                if bh_px > 0:
                                    bbox_aspect_ratios.append(bw_px / bh_px)
                                else:
                                    bbox_aspect_ratios.append(0.0)
                                    
                                class_counts[cls_id] += 1
                                class_bbox_rel_widths[cls_id].append(bw)
                                class_bbox_rel_heights[cls_id].append(bh)
                                class_bbox_abs_widths[cls_id].append(bw_px)
                                class_bbox_abs_heights[cls_id].append(bh_px)
                                
                                img_bboxes_count += 1
                                total_bboxes += 1
                            except ValueError:
                                continue
            
            bboxes_per_image.append(img_bboxes_count)
            if img_bboxes_count == 0:
                empty_images += 1
                
        except Exception as e:
            continue
            
    processed_count = len(img_widths)
    if processed_count == 0:
        return {"error": "No valid images could be read."}
        
    stats = {
        "num_classes": len(names_map),
        "class_names": list(names_map.values()),
        "total_images": len(image_paths),
        "processed_images": processed_count,
        "empty_images": empty_images,
        
        "image_size_stats": {
            "average_width": sum(img_widths) / processed_count,
            "average_height": sum(img_heights) / processed_count,
            "min_width": min(img_widths),
            "max_width": max(img_widths),
            "min_height": min(img_heights),
            "max_height": max(img_heights),
            "average_aspect_ratio": sum(img_aspect_ratios) / processed_count
        },
        
        "bbox_stats": {
            "total_bboxes": total_bboxes,
            "average_bboxes_per_image": total_bboxes / processed_count if processed_count > 0 else 0.0,
            "min_bboxes_per_image": min(bboxes_per_image) if bboxes_per_image else 0,
            "max_bboxes_per_image": max(bboxes_per_image) if bboxes_per_image else 0,
            
            # Relative dimensions
            "average_relative_width": sum(bbox_rel_widths) / total_bboxes if total_bboxes > 0 else 0.0,
            "average_relative_height": sum(bbox_rel_heights) / total_bboxes if total_bboxes > 0 else 0.0,
            "min_relative_width": min(bbox_rel_widths) if bbox_rel_widths else 0.0,
            "max_relative_width": max(bbox_rel_widths) if bbox_rel_widths else 0.0,
            "min_relative_height": min(bbox_rel_heights) if bbox_rel_heights else 0.0,
            "max_relative_height": max(bbox_rel_heights) if bbox_rel_heights else 0.0,
            
            # Absolute dimensions
            "average_absolute_width": sum(bbox_abs_widths) / total_bboxes if total_bboxes > 0 else 0.0,
            "average_absolute_height": sum(bbox_abs_heights) / total_bboxes if total_bboxes > 0 else 0.0,
            "min_absolute_width": min(bbox_abs_widths) if bbox_abs_widths else 0.0,
            "max_absolute_width": max(bbox_abs_widths) if bbox_abs_widths else 0.0,
            "min_absolute_height": min(bbox_abs_heights) if bbox_abs_heights else 0.0,
            "max_absolute_height": max(bbox_abs_heights) if bbox_abs_heights else 0.0,
            
            "average_aspect_ratio": sum(bbox_aspect_ratios) / total_bboxes if total_bboxes > 0 else 0.0
        },
        
        "class_distribution": {}
    }
    
    for cls_id, count in class_counts.items():
        cls_name = names_map.get(cls_id, f"class_{cls_id}")
        cls_rel_w = class_bbox_rel_widths[cls_id]
        cls_rel_h = class_bbox_rel_heights[cls_id]
        cls_abs_w = class_bbox_abs_widths[cls_id]
        cls_abs_h = class_bbox_abs_heights[cls_id]
        
        stats["class_distribution"][cls_name] = {
            "class_id": cls_id,
            "count": count,
            "percentage": (count / total_bboxes * 100) if total_bboxes > 0 else 0.0,
            "avg_relative_width": sum(cls_rel_w) / count if count > 0 else 0.0,
            "avg_relative_height": sum(cls_rel_h) / count if count > 0 else 0.0,
            "avg_absolute_width": sum(cls_abs_w) / count if count > 0 else 0.0,
            "avg_absolute_height": sum(cls_abs_h) / count if count > 0 else 0.0
        }
        
    return stats

