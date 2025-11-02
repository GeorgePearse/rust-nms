#!/usr/bin/env python3
"""
Reduce COCO dataset to 10% of original size.

Creates a valid COCO JSON file with 10% of images and their annotations.
Maintains full structure: images, annotations, categories, info, licenses.
"""

import json
import random
from pathlib import Path


def reduce_coco(input_path: Path, output_path: Path, sample_rate: float = 0.1):
    """
    Reduce COCO dataset by sampling images.

    Args:
        input_path: Path to original COCO JSON
        output_path: Path to save reduced COCO JSON
        sample_rate: Fraction of images to keep (default: 0.1 for 10%)
    """
    print(f"Loading COCO annotations from {input_path}...")
    with open(input_path, 'r') as f:
        coco = json.load(f)

    # Get total counts
    total_images = len(coco['images'])
    total_annotations = len(coco['annotations'])

    print(f"Original COCO:")
    print(f"  Images: {total_images:,}")
    print(f"  Annotations: {total_annotations:,}")
    print(f"  Categories: {len(coco['categories'])}")

    # Sample images
    random.seed(42)  # For reproducibility
    n_sample = int(total_images * sample_rate)
    sampled_images = random.sample(coco['images'], n_sample)
    sampled_image_ids = {img['id'] for img in sampled_images}

    print(f"\nSampling {sample_rate*100}% of images ({n_sample:,} images)...")

    # Filter annotations to only those belonging to sampled images
    sampled_annotations = [
        ann for ann in coco['annotations']
        if ann['image_id'] in sampled_image_ids
    ]

    # Create reduced COCO structure
    reduced_coco = {
        'info': coco.get('info', {}),
        'licenses': coco.get('licenses', []),
        'categories': coco['categories'],  # Keep all categories
        'images': sampled_images,
        'annotations': sampled_annotations,
    }

    print(f"\nReduced COCO:")
    print(f"  Images: {len(reduced_coco['images']):,}")
    print(f"  Annotations: {len(reduced_coco['annotations']):,}")
    print(f"  Categories: {len(reduced_coco['categories'])}")
    print(f"  Size reduction: {(1 - len(sampled_annotations)/total_annotations)*100:.1f}%")

    # Save reduced dataset
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(reduced_coco, f)

    # Check file sizes
    original_size = input_path.stat().st_size / (1024 * 1024)
    reduced_size = output_path.stat().st_size / (1024 * 1024)

    print(f"\nFile sizes:")
    print(f"  Original: {original_size:.1f} MB")
    print(f"  Reduced:  {reduced_size:.1f} MB ({reduced_size/original_size*100:.1f}%)")
    print(f"\n✓ Reduced COCO dataset saved!")


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    cache_dir = script_dir / "coco_cache"

    input_file = cache_dir / "instances_train2017.json"
    output_file = cache_dir / "instances_train2017_10pct.json"

    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        print("Please run run_benchmarks.py first to download COCO annotations.")
        return 1

    reduce_coco(input_file, output_file, sample_rate=0.1)
    return 0


if __name__ == "__main__":
    exit(main())
