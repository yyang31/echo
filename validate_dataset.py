#!/usr/bin/env python3
"""
Validate the YOLO dataset before training.

Checks for:
- Missing or corrupted images
- Invalid label formats
- Out-of-range bounding boxes
- Mismatched image/label counts
- Invalid class IDs
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np

def validate_dataset(dataset_dir: Path) -> bool:
    """Validate the entire YOLO dataset. Returns True if valid, False otherwise."""
    
    issues_found = False
    
    for split in ["train", "val"]:
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"ERROR: Missing {split} split directories")
            return False
        
        print(f"\nValidating {split} split...")
        
        image_files = sorted(images_dir.glob("*"))
        label_files = sorted(labels_dir.glob("*.txt"))
        
        # Check counts
        img_count = len(image_files)
        lbl_count = len(label_files)
        print(f"  Images: {img_count}, Labels: {lbl_count}")
        
        if img_count != lbl_count:
            print(f"  WARNING: Image/label count mismatch!")
            issues_found = True
        
        # Validate each image-label pair
        corrupted_images = []
        invalid_labels = []
        bbox_issues = []
        
        for img_file in image_files:
            # Check image
            try:
                with Image.open(img_file) as img:
                    w, h = img.size
                    if w < 32 or h < 32:
                        corrupted_images.append(f"{img_file.name}: dimension too small ({w}x{h})")
                    if w > 10000 or h > 10000:
                        corrupted_images.append(f"{img_file.name}: dimension too large ({w}x{h})")
            except Exception as e:
                corrupted_images.append(f"{img_file.name}: {str(e)}")
                continue
            
            # Check label
            lbl_file = labels_dir / img_file.with_suffix(".txt").name
            if not lbl_file.exists():
                invalid_labels.append(f"{img_file.name}: missing label file")
                continue
            
            try:
                with open(lbl_file, 'r') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 5:
                        invalid_labels.append(f"{img_file.name}:{line_num}: invalid format ({len(parts)} tokens)")
                        continue
                    
                    try:
                        class_id = int(float(parts[0]))
                        x = float(parts[1])
                        y = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                        
                        # Check for NaN or Inf
                        if not all(np.isfinite([x, y, w, h])):
                            bbox_issues.append(f"{img_file.name}:{line_num}: NaN/Inf values")
                            continue
                        
                        # Check valid ranges
                        if class_id < 0 or class_id > 2:
                            bbox_issues.append(f"{img_file.name}:{line_num}: class_id {class_id} out of range [0,2]")
                        
                        if x < 0 or x > 1 or y < 0 or y > 1:
                            bbox_issues.append(f"{img_file.name}:{line_num}: center coords out of [0,1]")
                        
                        if w <= 0 or w > 1 or h <= 0 or h > 1:
                            bbox_issues.append(f"{img_file.name}:{line_num}: bbox size out of (0,1]")
                    
                    except (ValueError, IndexError) as e:
                        invalid_labels.append(f"{img_file.name}:{line_num}: parse error - {str(e)}")
                        
            except Exception as e:
                invalid_labels.append(f"{img_file.name}: label read error - {str(e)}")
        
        # Report issues
        if corrupted_images:
            print(f"  Found {len(corrupted_images)} corrupted images:")
            for issue in corrupted_images[:5]:
                print(f"    - {issue}")
            if len(corrupted_images) > 5:
                print(f"    ... and {len(corrupted_images) - 5} more")
            issues_found = True
        
        if invalid_labels:
            print(f"  Found {len(invalid_labels)} invalid labels:")
            for issue in invalid_labels[:5]:
                print(f"    - {issue}")
            if len(invalid_labels) > 5:
                print(f"    ... and {len(invalid_labels) - 5} more")
            issues_found = True
        
        if bbox_issues:
            print(f"  Found {len(bbox_issues)} bbox issues:")
            for issue in bbox_issues[:5]:
                print(f"    - {issue}")
            if len(bbox_issues) > 5:
                print(f"    ... and {len(bbox_issues) - 5} more")
            issues_found = True
        
        if not (corrupted_images or invalid_labels or bbox_issues):
            print(f"  ✓ {split} split validation passed")
    
    return not issues_found


if __name__ == "__main__":
    dataset_dir = Path("yolo/door_window_stair_yolo_dataset")
    
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found at {dataset_dir}")
        sys.exit(1)
    
    print(f"Validating dataset at {dataset_dir.resolve()}")
    
    if validate_dataset(dataset_dir):
        print("\n✓ Dataset validation passed!")
        sys.exit(0)
    else:
        print("\n✗ Dataset validation found issues!")
        sys.exit(1)
