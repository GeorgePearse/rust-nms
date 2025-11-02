"""
Test script for rust_nms Python bindings
"""
import numpy as np


def test_nms():
    """Test basic NMS functionality"""
    try:
        import rust_nms
    except ImportError:
        print("rust_nms not installed. Run: maturin develop")
        return

    # Create test data: overlapping boxes
    boxes = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 1.0, 11.0, 11.0],
        [50.0, 50.0, 60.0, 60.0],
        [51.0, 51.0, 61.0, 61.0],
    ], dtype=np.float32)

    scores = np.array([0.9, 0.8, 0.95, 0.85], dtype=np.float32)

    # Run NMS
    keep_indices = rust_nms.nms(boxes, scores, iou_threshold=0.5)

    print(f"Original boxes: {len(boxes)}")
    print(f"After NMS: {len(keep_indices)}")
    print(f"Kept indices: {keep_indices}")
    print(f"Kept boxes:\n{boxes[keep_indices]}")
    print(f"Kept scores: {scores[keep_indices]}")

    assert len(keep_indices) == 2, "Should keep 2 boxes"
    assert 0 in keep_indices or 1 in keep_indices, "Should keep one from first cluster"
    assert 2 in keep_indices or 3 in keep_indices, "Should keep one from second cluster"

    print("✓ NMS test passed!")


def test_mask_to_polygons():
    """Test soft mask to polygon conversion"""
    try:
        import rust_nms
    except ImportError:
        print("rust_nms not installed. Run: maturin develop")
        return

    # Create a test mask with two squares
    mask = np.zeros((20, 20), dtype=np.float32)
    mask[5:10, 5:10] = 1.0  # First square
    mask[12:18, 12:18] = 0.9  # Second square

    # Add some noise
    mask[2, 2] = 0.3  # Below threshold, should be ignored

    # Convert to polygons
    polygons = rust_nms.mask_to_polygons(mask, threshold=0.5, min_area=5)

    print(f"\nFound {len(polygons)} polygons")
    for i, poly in enumerate(polygons):
        print(f"  Polygon {i+1}: {len(poly)} points")
        if len(poly) <= 10:
            print(f"    Points: {poly}")

    assert len(polygons) >= 1, "Should find at least one polygon"

    print("✓ Mask to polygons test passed!")


def benchmark_nms():
    """Benchmark NMS performance"""
    try:
        import rust_nms
        import time
    except ImportError:
        print("rust_nms not installed. Run: maturin develop")
        return

    # Generate random boxes
    n_boxes = 10000
    np.random.seed(42)

    boxes = np.random.rand(n_boxes, 4).astype(np.float32) * 100
    # Ensure x2 > x1 and y2 > y1
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]

    scores = np.random.rand(n_boxes).astype(np.float32)

    # Benchmark
    start = time.time()
    keep_indices = rust_nms.nms(boxes, scores, iou_threshold=0.5)
    elapsed = time.time() - start

    print(f"\nBenchmark: {n_boxes} boxes")
    print(f"  Time: {elapsed*1000:.2f}ms")
    print(f"  Kept: {len(keep_indices)} boxes")
    print(f"  Throughput: {n_boxes/elapsed:.0f} boxes/sec")


def visualize_nms():
    """Visualize NMS results (requires matplotlib)"""
    try:
        import rust_nms
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return

    # Create test data
    boxes = np.array([
        [10, 10, 50, 50],
        [15, 15, 55, 55],
        [20, 20, 60, 60],
        [100, 100, 150, 150],
        [105, 105, 155, 155],
    ], dtype=np.float32)

    scores = np.array([0.9, 0.85, 0.8, 0.95, 0.88], dtype=np.float32)

    # Run NMS
    keep_indices = rust_nms.nms(boxes, scores, iou_threshold=0.3)

    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Before NMS
    ax1.set_title("Before NMS")
    ax1.set_xlim(0, 200)
    ax1.set_ylim(0, 200)
    ax1.set_aspect('equal')
    for i, box in enumerate(boxes):
        rect = patches.Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
            linewidth=2, edgecolor='red', facecolor='none', alpha=0.6
        )
        ax1.add_patch(rect)
        ax1.text(box[0], box[1]-5, f'{scores[i]:.2f}', fontsize=10)

    # After NMS
    ax2.set_title("After NMS")
    ax2.set_xlim(0, 200)
    ax2.set_ylim(0, 200)
    ax2.set_aspect('equal')
    for idx in keep_indices:
        box = boxes[idx]
        rect = patches.Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
            linewidth=2, edgecolor='green', facecolor='none', alpha=0.8
        )
        ax2.add_patch(rect)
        ax2.text(box[0], box[1]-5, f'{scores[idx]:.2f}', fontsize=10)

    plt.tight_layout()
    plt.savefig('/Users/georgepearse/Documents/GitHub/rust-nms/nms_visualization.png', dpi=150)
    print("\n✓ Visualization saved to nms_visualization.png")


def visualize_mask_to_polygons():
    """Visualize mask to polygon conversion (requires matplotlib)"""
    try:
        import rust_nms
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return

    # Create a more complex mask
    mask = np.zeros((100, 100), dtype=np.float32)

    # Create a circle
    y, x = np.ogrid[:100, :100]
    circle_mask = ((x - 30)**2 + (y - 30)**2 <= 15**2)
    mask[circle_mask] = 1.0

    # Create a rectangle
    mask[60:80, 20:70] = 0.9

    # Create a soft gradient region
    xx, yy = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
    mask[70:90, 70:90] = (xx + yy) / 2

    # Convert to polygons
    polygons = rust_nms.mask_to_polygons(mask, threshold=0.5, min_area=10)

    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Original mask
    ax1.set_title("Original Soft Mask")
    ax1.imshow(mask, cmap='viridis', origin='lower')
    ax1.set_aspect('equal')
    plt.colorbar(ax1.images[0], ax=ax1, label='Score')

    # Extracted polygons
    ax2.set_title(f"Extracted Polygons (n={len(polygons)})")
    ax2.imshow(mask > 0.5, cmap='gray', origin='lower', alpha=0.3)
    ax2.set_aspect('equal')

    for poly in polygons:
        if poly:
            poly_array = np.array(poly)
            ax2.plot(poly_array[:, 0], poly_array[:, 1], 'r-', linewidth=2)
            ax2.scatter(poly_array[:, 0], poly_array[:, 1], c='blue', s=10)

    plt.tight_layout()
    plt.savefig('/Users/georgepearse/Documents/GitHub/rust-nms/polygon_visualization.png', dpi=150)
    print("\n✓ Polygon visualization saved to polygon_visualization.png")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing rust_nms Python bindings")
    print("=" * 60)

    test_nms()
    test_mask_to_polygons()
    benchmark_nms()

    print("\n" + "=" * 60)
    print("Optional: Generating visualizations")
    print("=" * 60)
    try:
        visualize_nms()
        visualize_mask_to_polygons()
    except Exception as e:
        print(f"Visualization skipped: {e}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
