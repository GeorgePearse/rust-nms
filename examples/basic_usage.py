"""
Basic usage examples for rust_nms
"""
import numpy as np

try:
    import rust_nms
except ImportError:
    print("Error: rust_nms not installed")
    print("Run: maturin develop --release")
    exit(1)


def example_nms():
    """Example: Non-Maximum Suppression"""
    print("=" * 60)
    print("Example 1: Non-Maximum Suppression")
    print("=" * 60)

    # Create overlapping bounding boxes
    # Format: [x1, y1, x2, y2]
    boxes = np.array([
        [10, 10, 50, 50],    # Box 1
        [15, 15, 55, 55],    # Box 2 (overlaps with 1)
        [20, 20, 60, 60],    # Box 3 (overlaps with 1 and 2)
        [100, 100, 150, 150],  # Box 4 (separate)
    ], dtype=np.float32)

    # Confidence scores for each box
    scores = np.array([0.9, 0.85, 0.8, 0.95], dtype=np.float32)

    print(f"\nInput: {len(boxes)} boxes")
    for i, (box, score) in enumerate(zip(boxes, scores)):
        print(f"  Box {i}: {box} score={score:.2f}")

    # Run NMS with IoU threshold of 0.3
    keep = rust_nms.nms(boxes, scores, iou_threshold=0.3)

    print(f"\nOutput: {len(keep)} boxes kept (indices: {keep})")
    for idx in keep:
        print(f"  Box {idx}: {boxes[idx]} score={scores[idx]:.2f}")

    print("\nExplanation:")
    print("  - Boxes 0, 1, 2 overlap significantly")
    print("  - Box 0 has highest score (0.9) among overlapping boxes")
    print("  - Box 3 is kept because it doesn't overlap with others")
    print("  - Box 4 has highest score (0.95) and is kept")


def example_mask_to_polygons():
    """Example: Soft Mask to Polygon Conversion"""
    print("\n" + "=" * 60)
    print("Example 2: Soft Mask to Polygon Conversion")
    print("=" * 60)

    # Create a soft mask with two regions
    height, width = 50, 50
    mask = np.zeros((height, width), dtype=np.float32)

    # Region 1: High confidence square
    mask[10:20, 10:20] = 1.0

    # Region 2: Medium confidence rectangle
    mask[30:45, 5:25] = 0.7

    # Add some low confidence noise (will be filtered out)
    mask[5, 5] = 0.3

    print(f"\nInput: {height}x{width} soft mask")
    print(f"  High confidence region: 10x10 square at (10, 10)")
    print(f"  Medium confidence region: 15x20 rectangle at (30, 5)")
    print(f"  Low confidence noise: single pixel at (5, 5)")

    # Convert to polygons with threshold=0.5
    polygons = rust_nms.mask_to_polygons(
        mask,
        threshold=0.5,
        min_area=5
    )

    print(f"\nOutput: {len(polygons)} polygons found")
    for i, poly in enumerate(polygons):
        print(f"  Polygon {i+1}: {len(poly)} vertices")

        # Calculate bounding box
        poly_array = np.array(poly)
        x_min, y_min = poly_array.min(axis=0)
        x_max, y_max = poly_array.max(axis=0)
        print(f"    Bounding box: ({x_min:.0f}, {y_min:.0f}) to ({x_max:.0f}, {y_max:.0f})")

    print("\nExplanation:")
    print("  - Threshold 0.5 filters out the noise (0.3 < 0.5)")
    print("  - Both regions pass threshold (1.0 and 0.7 >= 0.5)")
    print("  - Each region becomes a separate polygon")


def example_real_world():
    """Example: Simulated object detection pipeline"""
    print("\n" + "=" * 60)
    print("Example 3: Simulated Object Detection Pipeline")
    print("=" * 60)

    # Simulate detections from an object detector
    # In reality, these would come from YOLO, Faster R-CNN, etc.
    np.random.seed(42)

    # Generate random detections
    n_detections = 100
    boxes = np.random.rand(n_detections, 4).astype(np.float32) * 200

    # Ensure x2 > x1 and y2 > y1
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]

    # Random confidence scores
    scores = np.random.rand(n_detections).astype(np.float32)

    print(f"\nRaw detections: {len(boxes)} boxes")
    print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")

    # Apply confidence threshold
    conf_threshold = 0.5
    mask = scores >= conf_threshold
    boxes_filtered = boxes[mask]
    scores_filtered = scores[mask]

    print(f"\nAfter confidence threshold ({conf_threshold}): {len(boxes_filtered)} boxes")

    # Apply NMS
    keep = rust_nms.nms(boxes_filtered, scores_filtered, iou_threshold=0.5)
    final_boxes = boxes_filtered[keep]
    final_scores = scores_filtered[keep]

    print(f"After NMS: {len(final_boxes)} boxes")
    print(f"  Reduction: {len(boxes)} → {len(final_boxes)} ({100*len(final_boxes)/len(boxes):.1f}%)")

    # Show top 5 detections
    top_k = min(5, len(final_scores))
    top_indices = np.argsort(final_scores)[::-1][:top_k]

    print(f"\nTop {top_k} detections:")
    for i, idx in enumerate(top_indices):
        box = final_boxes[idx]
        score = final_scores[idx]
        print(f"  {i+1}. Box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}] "
              f"Score: {score:.3f}")


if __name__ == "__main__":
    example_nms()
    example_mask_to_polygons()
    example_real_world()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Run tests: python test_nms.py")
    print("  - Check visualizations (requires matplotlib)")
    print("  - Try with your own data!")
