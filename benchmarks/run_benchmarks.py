#!/usr/bin/env python3
"""
Benchmark runner for rust-nms performance tracking.

This script runs benchmarks for both NMS and mask_to_polygons operations
using real COCO dataset annotations for realistic testing.
"""

import time
import json
import sys
import os
import subprocess
import urllib.request
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

import numpy as np

# Add parent directory to path to import rust_nms
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import rust_nms
except ImportError:
    print("Error: rust_nms module not found. Please build with 'maturin develop --release'")
    sys.exit(1)


COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_ANNOTATIONS_FILE = "instances_train2017.json"
COCO_REDUCED_FILE = "instances_train2017_10pct.json"  # 10% reduced dataset


def download_coco_annotations(cache_dir: Path) -> Path:
    """
    Download COCO annotations if not already cached.

    Args:
        cache_dir: Directory to cache annotations

    Returns:
        Path to the annotations JSON file
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = cache_dir / COCO_ANNOTATIONS_FILE

    if annotations_path.exists():
        print(f"Using cached COCO annotations: {annotations_path}")
        return annotations_path

    print("Downloading COCO annotations (first run only, ~250MB)...")
    zip_path = cache_dir / "annotations_trainval2017.zip"

    # Download
    urllib.request.urlretrieve(COCO_ANNOTATIONS_URL, zip_path)
    print("Download complete. Extracting...")

    # Extract
    import zipfile
    import shutil
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(cache_dir)

    # Move from annotations/ subdirectory
    extracted_path = cache_dir / "annotations" / COCO_ANNOTATIONS_FILE
    if extracted_path.exists():
        extracted_path.rename(annotations_path)
        # Clean up annotations directory
        shutil.rmtree(cache_dir / "annotations", ignore_errors=True)

    zip_path.unlink()
    print(f"COCO annotations ready: {annotations_path}")

    return annotations_path


def create_reduced_coco(cache_dir: Path, sample_rate: float = 0.1) -> Path:
    """
    Create a reduced COCO dataset (10% of images).

    Args:
        cache_dir: Directory containing COCO annotations
        sample_rate: Fraction of images to keep (default: 0.1)

    Returns:
        Path to reduced COCO JSON file
    """
    import random

    full_path = cache_dir / COCO_ANNOTATIONS_FILE
    reduced_path = cache_dir / COCO_REDUCED_FILE

    if reduced_path.exists():
        print(f"Using existing reduced COCO dataset: {reduced_path}")
        return reduced_path

    print(f"Creating reduced COCO dataset ({sample_rate*100}% of images)...")

    # Load full COCO
    with open(full_path, 'r') as f:
        coco = json.load(f)

    # Sample images
    random.seed(42)
    n_sample = int(len(coco['images']) * sample_rate)
    sampled_images = random.sample(coco['images'], n_sample)
    sampled_image_ids = {img['id'] for img in sampled_images}

    # Filter annotations
    sampled_annotations = [
        ann for ann in coco['annotations']
        if ann['image_id'] in sampled_image_ids
    ]

    # Create reduced structure
    reduced_coco = {
        'info': coco.get('info', {}),
        'licenses': coco.get('licenses', []),
        'categories': coco['categories'],
        'images': sampled_images,
        'annotations': sampled_annotations,
    }

    # Save
    with open(reduced_path, 'w') as f:
        json.dump(reduced_coco, f)

    print(f"✓ Created reduced dataset: {len(sampled_images):,} images, {len(sampled_annotations):,} annotations")

    return reduced_path


def load_coco_boxes_and_scores(annotations_path: Path, max_boxes: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load bounding boxes and scores from COCO annotations.

    Args:
        annotations_path: Path to COCO JSON file
        max_boxes: Maximum number of boxes to load (None for all)

    Returns:
        Tuple of (boxes, scores) as numpy arrays
    """
    print(f"Loading COCO annotations from {annotations_path}...")

    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)

    annotations = coco_data['annotations']
    if max_boxes:
        annotations = annotations[:max_boxes]

    boxes = []
    scores = []

    for ann in annotations:
        # COCO format: [x, y, width, height]
        # Convert to [x1, y1, x2, y2]
        x, y, w, h = ann['bbox']
        boxes.append([x, y, x + w, y + h])

        # Use area as a proxy for confidence score (normalized)
        # In real detection, this would be the model's confidence
        scores.append(min(1.0, ann['area'] / 10000.0))

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    print(f"Loaded {len(boxes):,} boxes from COCO dataset")

    return boxes, scores


def get_git_info() -> Dict[str, str]:
    """Get current git metadata."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        return {"commit": commit, "branch": branch}
    except subprocess.CalledProcessError:
        return {"commit": "unknown", "branch": "unknown"}


def benchmark_nms_coco(boxes: np.ndarray, scores: np.ndarray, n_boxes: int, n_runs: int = 5) -> Dict[str, float]:
    """
    Benchmark NMS operation using COCO data.

    Args:
        boxes: Full COCO boxes array
        scores: Full COCO scores array
        n_boxes: Number of boxes to test
        n_runs: Number of runs to average over

    Returns:
        Dictionary with timing statistics
    """
    # Sample n_boxes from the dataset
    if n_boxes > len(boxes):
        n_boxes = len(boxes)

    indices = np.random.choice(len(boxes), size=n_boxes, replace=False)
    test_boxes = boxes[indices].copy()
    test_scores = scores[indices].copy()

    # Warm-up run
    _ = rust_nms.nms(test_boxes, test_scores, iou_threshold=0.5)

    # Benchmark runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        keep_indices = rust_nms.nms(test_boxes, test_scores, iou_threshold=0.5)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times = np.array(times)
    avg_time = np.mean(times)

    return {
        "time_ms": avg_time * 1000,
        "time_std_ms": np.std(times) * 1000,
        "throughput_boxes_per_sec": n_boxes / avg_time,
        "n_boxes": n_boxes,
        "n_kept": len(keep_indices),
        "n_runs": n_runs,
        "dataset": "coco_train2017",
    }


def benchmark_mask_to_polygons(size: int, n_runs: int = 5) -> Dict[str, float]:
    """
    Benchmark mask_to_polygons operation.

    Args:
        size: Size of square mask (size x size)
        n_runs: Number of runs to average over

    Returns:
        Dictionary with timing statistics
    """
    np.random.seed(42)

    # Generate random soft mask with some structure
    mask = np.random.rand(size, size).astype(np.float32)
    # Add some blobs to make it more realistic
    for _ in range(10):
        cx, cy = np.random.randint(0, size, 2)
        radius = np.random.randint(20, 100)
        y, x = np.ogrid[-cy:size-cy, -cx:size-cx]
        circle = x*x + y*y <= radius*radius
        mask[circle] = np.random.rand()

    # Warm-up run
    _ = rust_nms.mask_to_polygons(mask, threshold=0.5, min_area=10)

    # Benchmark runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        polygons = rust_nms.mask_to_polygons(mask, threshold=0.5, min_area=10)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times = np.array(times)
    avg_time = np.mean(times)

    return {
        "time_ms": avg_time * 1000,
        "time_std_ms": np.std(times) * 1000,
        "throughput_megapixels_per_sec": (size * size / 1e6) / avg_time,
        "size": f"{size}x{size}",
        "n_polygons": len(polygons),
        "n_runs": n_runs,
    }


def run_all_benchmarks() -> Dict[str, Any]:
    """Run all benchmarks and return results."""
    print("Running benchmarks with COCO dataset (10% subset)...")
    print("=" * 60)

    # Setup COCO data
    cache_dir = Path(__file__).parent / "coco_cache"

    # Download full COCO first
    full_annotations_path = download_coco_annotations(cache_dir)

    # Create reduced dataset (10%)
    reduced_annotations_path = create_reduced_coco(cache_dir, sample_rate=0.1)

    # Use reduced dataset for benchmarks
    all_boxes, all_scores = load_coco_boxes_and_scores(reduced_annotations_path)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **get_git_info(),
        "benchmarks": {},
    }

    # NMS benchmarks with COCO data
    # Adjust sizes for 10% dataset (originally ~860k boxes, now ~86k)
    max_boxes = len(all_boxes)
    nms_sizes = [1000, 10000, max_boxes]  # Test with available data

    for n_boxes in nms_sizes:
        if n_boxes > max_boxes:
            continue

        print(f"\nBenchmarking NMS with {n_boxes:,} COCO boxes...")
        result = benchmark_nms_coco(all_boxes, all_scores, n_boxes)
        results["benchmarks"][f"nms_{n_boxes}"] = result
        print(f"  Time: {result['time_ms']:.2f} ± {result['time_std_ms']:.2f} ms")
        print(f"  Throughput: {result['throughput_boxes_per_sec']:,.0f} boxes/sec")
        print(f"  Kept: {result['n_kept']} boxes")

    # Mask to polygons benchmarks
    mask_sizes = [256, 512, 1024]
    for size in mask_sizes:
        print(f"\nBenchmarking mask_to_polygons with {size}x{size} mask...")
        result = benchmark_mask_to_polygons(size)
        results["benchmarks"][f"mask_to_polygons_{size}"] = result
        print(f"  Time: {result['time_ms']:.2f} ± {result['time_std_ms']:.2f} ms")
        print(f"  Throughput: {result['throughput_megapixels_per_sec']:.1f} megapixels/sec")
        print(f"  Polygons: {result['n_polygons']}")

    print("\n" + "=" * 60)
    print("Benchmarks complete!")

    return results
    for size in mask_sizes:
        print(f"\nBenchmarking mask_to_polygons with {size}x{size} mask...")
        result = benchmark_mask_to_polygons(size)
        results["benchmarks"][f"mask_to_polygons_{size}"] = result
        print(f"  Time: {result['time_ms']:.2f} ± {result['time_std_ms']:.2f} ms")
        print(f"  Throughput: {result['throughput_megapixels_per_sec']:.1f} megapixels/sec")
        print(f"  Polygons: {result['n_polygons']}")

    print("\n" + "=" * 60)
    print("Benchmarks complete!")

    return results


def main():
    """Main entry point."""
    results = run_all_benchmarks()

    # Output JSON to stdout
    print("\nJSON Output:")
    print(json.dumps(results, indent=2))

    # Also save to file
    output_file = os.path.join(
        os.path.dirname(__file__), "latest_benchmark.json"
    )
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
