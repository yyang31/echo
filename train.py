#!/usr/bin/env python3
"""
Train YOLO26n on door/window/stair detection dataset.

Requirements:
    pip install ultralytics pillow numpy

Usage:
    python3 train.py                    # Use subset for faster training
    python3 train.py --fresh            # Start fresh training
    python3 train.py --full-training    # Use full dataset with 80/20 split

Outputs:
    ./yolo/door_window_stair_yolo_dataset/
    ./yolo/runs/door_window_stair_yolo/
"""

import shutil
import argparse
from pathlib import Path
import random

from ultralytics import YOLO


# -------------------------
# Config
# -------------------------
ROOT = Path(".").resolve()
DATASET_DIR = ROOT / "yolo" / "door_window_stair_yolo_dataset"
SOURCE_DATA_DIR = ROOT / "door_window_stair_dataset"
RUN_DIR = ROOT / "yolo" / "runs" / "door_window_stair_yolo"

# Training limits (for quick testing; --full-training uses all data with 80/20 split)
TRAIN_LIMIT = 500    # Quick training limit
VAL_LIMIT = 125      # Quick validation limit
TRAIN_EPOCHS = 20
IMG_EXT = ".jpg"

CLASS_NAMES = ["door", "stairs", "window"]


# -------------------------
# Helpers
# -------------------------
def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def make_dirs():
    for split in ["train", "val"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def get_image_files(data_dir: Path):
    """Get all .jpg and .JPG files in the data directory."""
    files = sorted(data_dir.glob("*.jpg")) + sorted(data_dir.glob("*.JPG"))
    return files


def get_device():
    """Auto-detect the best available device."""
    try:
        import torch
        if torch.backends.mps.is_available():
            print("Device: Apple Silicon (MPS)")
            return "mps"
        elif torch.cuda.is_available():
            print("Device: NVIDIA GPU (CUDA)")
            return "cuda"
    except Exception as e:
        print(f"GPU detection failed: {e}")
    
    print("Device: CPU")
    return "cpu"


def copy_sample(img_path: Path, lbl_path: Path, out_img_dir: Path, out_lbl_dir: Path):
    """Copy image and label files to output directories.

    Perform basic validation on the label file: non-empty, each line
    contains at least 5 tokens (class + 4 bbox values), class index is
    within range of `CLASS_NAMES`, and bbox values parse as floats.
    If validation fails the image is skipped to avoid training crashes.
    """
    if not lbl_path.exists():
        print(f"Skipping {img_path.name} - label file not found")
        return False

    # Read and validate label content
    try:
        with open(lbl_path, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    except Exception as e:
        print(f"Skipping {img_path.name} - failed reading label file: {e}")
        return False

    if not lines:
        print(f"Skipping {img_path.name} - empty label file")
        return False

    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 5:
            print(f"Skipping {img_path.name} - invalid label line {i+1}: '{line}'")
            return False
        # Parse class index
        try:
            cls = int(float(parts[0]))
        except Exception:
            print(f"Skipping {img_path.name} - invalid class id in line {i+1}: '{parts[0]}'")
            return False
        if cls < 0 or cls >= len(CLASS_NAMES):
            print(f"Skipping {img_path.name} - class index {cls} out of range (0..{len(CLASS_NAMES)-1})")
            return False
        # Parse bbox values
        try:
            _ = list(map(float, parts[1:5]))
        except Exception:
            print(f"Skipping {img_path.name} - invalid bbox values in line {i+1}")
            return False

    # Passed validation, copy files
    shutil.copy2(img_path, out_img_dir / img_path.name)
    shutil.copy2(lbl_path, out_lbl_dir / lbl_path.name)
    return True


def build_split_from_existing(source_dir: Path, full_training: bool):
    """Build train/val split from existing dataset."""
    img_files = get_image_files(source_dir)
    
    if not img_files:
        raise ValueError(f"No .jpg files found in {source_dir}")
    
    print(f"Found {len(img_files)} images in {source_dir}")
    random.shuffle(img_files)
    
    # Determine split
    if full_training:
        # Use all data with 80/20 split
        num_train = int(len(img_files) * 0.8)
        print(f"Using full dataset: {num_train} train, {len(img_files) - num_train} val")
    else:
        # Use limits for faster training
        num_train = min(len(img_files), TRAIN_LIMIT)
        remaining = len(img_files) - num_train
        num_val = min(remaining, VAL_LIMIT)
        print(f"Using limited dataset: up to {num_train} train, up to {num_val} val")
        img_files = img_files[:num_train + num_val]

    # Split after shuffling so both subsets are representative
    num_train = int(len(img_files) * 0.8) if full_training else min(len(img_files), TRAIN_LIMIT)
    train_files = img_files[:num_train]
    val_files = img_files[num_train:]
    
    # Copy files
    train_img_dir = DATASET_DIR / "images" / "train"
    train_lbl_dir = DATASET_DIR / "labels" / "train"
    val_img_dir = DATASET_DIR / "images" / "val"
    val_lbl_dir = DATASET_DIR / "labels" / "val"
    
    print(f"Copying training samples...")
    train_count = 0
    for img_path in train_files:
        lbl_path = img_path.with_suffix(".txt")
        if copy_sample(img_path, lbl_path, train_img_dir, train_lbl_dir):
            train_count += 1
    print(f"Copied {train_count} training samples")
    
    print(f"Copying validation samples...")
    val_count = 0
    for img_path in val_files:
        lbl_path = img_path.with_suffix(".txt")
        if copy_sample(img_path, lbl_path, val_img_dir, val_lbl_dir):
            val_count += 1
    print(f"Copied {val_count} validation samples")


def write_dataset_yaml():
    yaml_text = f"""path: {DATASET_DIR.resolve()}
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names:
"""
    for idx, name in enumerate(CLASS_NAMES):
        yaml_text += f"  {idx}: {name}\n"

    yaml_path = DATASET_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_text)

    return yaml_path

def train_yolo(data_yaml: Path):
    device = get_device()
    checkpoint = RUN_DIR / "weights" / "last.pt"
    if checkpoint.exists():
        print(f"Resuming training from {checkpoint}")
        model = YOLO(str(checkpoint))
        results = model.train(device=device, resume=True)
    else:
        model = YOLO("yolo26n.pt")
        results = model.train(
            data=str(data_yaml),
            epochs=TRAIN_EPOCHS,
            imgsz=640,
            batch=16,
            project=str(RUN_DIR.parent),
            name=RUN_DIR.name,
            exist_ok=True,
            pretrained=True,
            verbose=True,
            plots=True,
            device=device,
        )

    return model, results

def validate_yolo(model, data_yaml: Path):
    metrics = model.val(
        data=str(data_yaml),
        imgsz=640,
        split="val",
        project=str(RUN_DIR.parent),
        name=f"{RUN_DIR.name}_val",
        exist_ok=True,
        plots=True,
    )
    return metrics


def write_summary(metrics, full_training: bool):
    summary_path = RUN_DIR / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("Door/Window/Stair Detection YOLO Training Summary\n")
        f.write("=================================================\n\n")
        f.write("Dataset: Door/Window/Stair detection dataset\n")
        if full_training:
            f.write("Training Mode: Full dataset (80/20 split)\n")
        else:
            f.write(f"Training Mode: Limited dataset ({TRAIN_LIMIT} train, {VAL_LIMIT} val)\n")
        f.write("Classes: door, stairs, window\n\n")

        names_and_values = [
            ("metrics.box.map", getattr(metrics.box, "map", None)),
            ("metrics.box.map50", getattr(metrics.box, "map50", None)),
            ("metrics.box.map75", getattr(metrics.box, "map75", None)),
            ("metrics.box.mp", getattr(metrics.box, "mp", None)),
            ("metrics.box.mr", getattr(metrics.box, "mr", None)),
        ]

        for name, value in names_and_values:
            if value is not None:
                f.write(f"{name}: {value}\n")

        f.write("\nExpected YOLO result files in this run directory typically include:\n")
        f.write("- results.csv\n")
        f.write("- results.png\n")
        f.write("- confusion_matrix.png\n")
        f.write("- confusion_matrix_normalized.png\n")
        f.write("- F1_curve.png\n")
        f.write("- P_curve.png\n")
        f.write("- R_curve.png\n")
        f.write("- PR_curve.png\n")
        f.write("- weights/best.pt\n")
        f.write("- weights/last.pt\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO26n on door/window/stair detection")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any saved checkpoint and retrain from scratch",
    )
    parser.add_argument(
        "--full-training",
        action="store_true",
        help="Use full dataset with 80/20 split (default: use subset for faster training)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Preparing dataset directories...")
    ensure_clean_dir(DATASET_DIR)
    make_dirs()

    print("Loading existing dataset...")
    if not SOURCE_DATA_DIR.exists():
        raise ValueError(f"Source dataset not found at {SOURCE_DATA_DIR}")
    
    build_split_from_existing(SOURCE_DATA_DIR, args.full_training)

    # create a yaml containing the class labels
    data_yaml = write_dataset_yaml()
    print(f"Wrote dataset YAML: {data_yaml}")

    # if --fresh is specified, restart the training
    if args.fresh:
        print(f"Starting a fresh training run in {RUN_DIR}")
        if RUN_DIR.exists():
            shutil.rmtree(RUN_DIR)

    print("Training YOLO26n...")
    model, _ = train_yolo(data_yaml)

    print("Running validation...")
    metrics = validate_yolo(model, data_yaml)

    write_summary(metrics, args.full_training)

    print("\nDone.")
    print(f"Dataset saved to: {DATASET_DIR.resolve()}")
    print(f"Training results saved to: {RUN_DIR.resolve()}")
    print("\nLook for these files in the run directory:")
    print("  weights/best.pt")
    print("  weights/last.pt")
    print("  results.csv")
    print("  results.png")
    print("  PR_curve.png")
    print("  P_curve.png")
    print("  R_curve.png")
    print("  F1_curve.png")
    print("  confusion_matrix.png")
    print("  confusion_matrix_normalized.png")
    print("  summary.txt")


if __name__ == "__main__":
    main()
