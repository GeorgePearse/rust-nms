# rust-nms

Fast Rust implementation of Non-Maximum Suppression (NMS) and soft mask to polygon conversion with Python bindings via maturin.

## Features

- **Fast NMS**: Efficient Non-Maximum Suppression for bounding box filtering
- **Mask to Polygons**: Convert soft segmentation masks (0-1 scores) to polygon contours
- **Python Bindings**: Seamless integration with NumPy arrays
- **Zero-Copy**: Efficient memory usage with array views
- **Type-Safe**: Rust's type system ensures correctness

## Installation

### From PyPI (once published)

```bash
pip install rust-nms
```

### From Source

Requires Rust and maturin:

```bash
# Install maturin
pip install maturin

# Build and install in development mode
cd rust-nms
maturin develop --release

# Or build wheel
maturin build --release
pip install target/wheels/rust_nms-*.whl
```

## Usage

### Non-Maximum Suppression

```python
import numpy as np
import rust_nms

boxes = np.array([
    [0.0, 0.0, 10.0, 10.0],
    [1.0, 1.0, 11.0, 11.0],
    [50.0, 50.0, 60.0, 60.0],
], dtype=np.float32)

scores = np.array([0.9, 0.8, 0.95], dtype=np.float32)

keep_indices = rust_nms.nms(boxes, scores, iou_threshold=0.5)
filtered_boxes = boxes[keep_indices]
filtered_scores = scores[keep_indices]
```

### Soft Mask to Polygons

```python
import numpy as np
import rust_nms

mask = np.zeros((100, 100), dtype=np.float32)
mask[20:80, 20:80] = 0.9
mask[10:30, 10:30] = 0.6

polygons = rust_nms.mask_to_polygons(mask, threshold=0.5, min_area=10)

for poly in polygons:
    print(f"{len(poly)} points: {poly[:3]}...")
```

## API Reference

### `nms(boxes, scores, iou_threshold=0.5, max_detections=None)`

Non-Maximum Suppression for bounding boxes.

**Parameters:**
- `boxes`: `np.ndarray[float32]` - Shape (N, 4), format [x1, y1, x2, y2]
- `scores`: `np.ndarray[float32]` - Shape (N,), confidence scores
- `iou_threshold`: `float` - IoU threshold for suppression (default: 0.5)
- `max_detections`: `int` - Optional limit for maximum detections (default: None)

**Returns:**
- `np.ndarray[uint]` - Indices of boxes to keep

**⚠️ Important:** NMS operates on a **single batch** of boxes. For video processing or multi-image scenarios, you **must** call NMS separately for each frame/image. Mixing boxes from different frames will cause incorrect suppression across frames.

```python
# ✓ CORRECT: Process each frame separately
for frame_boxes, frame_scores in zip(all_frames_boxes, all_frames_scores):
    keep = rust_nms.nms(frame_boxes, frame_scores, iou_threshold=0.5)
    
# ✗ WRONG: Don't mix boxes from different frames!
all_boxes = np.concatenate(all_frames_boxes)
all_scores = np.concatenate(all_frames_scores)
keep = rust_nms.nms(all_boxes, all_scores, iou_threshold=0.5)  # INCORRECT!
```

### `mask_to_polygons(mask, threshold=0.5, min_area=10)`

Convert soft segmentation mask to polygon contours.

**Parameters:**
- `mask`: `np.ndarray[float32]` - Shape (H, W), values in [0.0, 1.0]
- `threshold`: `float` - Binarization threshold (default: 0.5)
- `min_area`: `int` - Minimum polygon area in pixels (default: 10)

**Returns:**
- `List[List[Tuple[float, float]]]` - List of polygons, each polygon is a list of (x, y) points

**Algorithm:**
1. Binarize mask at threshold
2. Find connected components via flood-fill
3. Extract contours using Moore-neighbor tracing
4. Filter by minimum area
5. Return polygon point lists

**Time Complexity:** O(H × W)

## Performance

Benchmarks on M1 MacBook Pro using COCO train2017 dataset:

| Operation | Input Size | Time | Throughput |
|-----------|-----------|------|------------|
| NMS | 10,000 boxes | ~15ms | ~660k boxes/sec |
| Mask to Polygons | 1024×1024 | ~5ms | ~200 megapixels/sec |

**📊 [View Performance Trends →](benchmarks/README.md)**

Performance is continuously tracked on every commit using real COCO annotations (10% subset). The CI automatically **fails if performance regresses >5%**, ensuring speed never degrades silently. Charts show time-series trends across different input sizes.

## Testing

Run Python tests:

```bash
python test_nms.py
```

Run Rust tests:

```bash
cargo test
```

## Benchmarking

Run Rust benchmarks:

```bash
# Option 1: Use the cargo alias (recommended)
cargo bench-rust

# Option 2: Explicitly disable default features
cargo bench --no-default-features
```

**Note:** The default `cargo bench` will fail with linking errors because the Python extension module features are incompatible with Rust-only benchmarks. Always use `--no-default-features` or the `bench-rust` alias when running benchmarks manually.

## Development Setup

### Pre-commit Performance Validation

This project uses Git pre-commit hooks to ensure code quality and performance:

**What it does:**
1. Runs all tests (`cargo test`)
2. Compares benchmarks with your previous commit (HEAD~1)
3. Blocks commits if tests fail or performance regresses

**Installation:**

```bash
./scripts/install-hooks.sh
```

**Requirements:**
- Every commit must pass all tests
- Every commit must show measurable performance improvement (any improvement >0%)
- Benchmarks are compared against the previous commit using `cargo bench`

**Bypass in emergencies:**

```bash
git commit --no-verify  # Not recommended
```

**How it works:**
- The hook runs `cargo bench --baseline previous` to compare your changes
- First commit in a branch skips comparison (no previous commit)
- Takes ~30-60 seconds per commit (runs benchmarks twice)
- Works with git worktrees

**Note:** This is a local development tool. The CI also validates performance using real COCO data and fails if performance regresses >5%.

## Project Structure

```
rust-nms/
├── src/
│   └── lib.rs              # Rust implementation + Python bindings
├── Cargo.toml              # Rust dependencies
├── pyproject.toml          # Python package metadata
├── test_nms.py             # Python tests and benchmarks
├── README.md               # This file
└── AGENTS.md               # Development references and citations
```

## Algorithm Details

### Non-Maximum Suppression

NMS eliminates redundant overlapping detections:

1. **Sort** by confidence score (highest first)
2. **Select** highest scoring box
3. **Suppress** overlapping boxes with IoU > threshold
4. **Repeat** until all boxes processed

**IoU (Intersection over Union):**

```
IoU = Area(A ∩ B) / Area(A ∪ B)
```

### Mask to Polygon Conversion

Converts probabilistic segmentation masks to vector polygons:

1. **Binarization**: Apply threshold to create binary mask
2. **Component Analysis**: Find connected regions via flood-fill
3. **Contour Tracing**: Extract boundaries using Moore-neighbor algorithm
4. **Filtering**: Remove small polygons below minimum area

**Moore-Neighbor Tracing:**
- Starts at boundary pixel
- Follows 8-connected neighbors clockwise
- Stops when returning to start point

## Implementation Philosophy

### Research-Oriented Development

This repository maintains **multiple NMS implementations** side-by-side, even when some are slower than others. This is an intentional design decision:

**Why keep old implementations?**

- 🔬 **Research baseline**: Older implementations provide benchmarking baselines to quantify improvements
- 📚 **Educational value**: Different optimization strategies demonstrate performance trade-offs
- 🧪 **Experimentation**: Alternative approaches may perform better on different hardware or workloads
- 📝 **Documentation**: Code serves as living documentation of what was tried and learned

**Where are implementations?**

- ⭐ **Primary/Production**: `src/lib.rs::nms_impl()` - The fastest, most optimized implementation
- 🔬 **Research variants**: `src/nms_impls.rs` - Alternative implementations kept for comparison

**Current optimizations in `nms_impl()`:**

The production implementation incorporates multiple performance optimizations discovered through systematic experimentation:

1. **Structure of Arrays (SoA)** layout for cache-friendly memory access
2. **Spatial indexing** with grid-based filtering (n > 500 boxes)
   - Reduces comparisons from O(n²) to O(n·k) where k is local density
   - Adaptive cell padding and intelligent fallback heuristics
3. **Immediate suppression marking** eliminates allocation overhead
4. **Manual loop unrolling** (blocks of 4) for instruction-level parallelism
5. **Unsafe operations** to eliminate bounds checks in hot paths
6. **Early rejection tests** before expensive IoU calculations

**Performance tracking:**

Every commit is automatically benchmarked. The codebase must show measurable performance improvement to merge - no regressions allowed. This ensures the production implementation always represents our best-known approach.

See [AGENTS.md](AGENTS.md) for detailed implementation notes and benchmarking requirements.

## Video & Multi-Frame Processing

**⚠️ Critical:** NMS must be applied **per-frame**. Never mix detections from different frames!

Boxes from different frames exist in different temporal contexts and should never suppress each other, even if they have identical coordinates. Always process each frame independently:

```python
import numpy as np
import rust_nms

# Example: Processing video frames from an object detector
video_detections = {
    0: {  # Frame 0
        'boxes': np.array([[10, 10, 50, 50], [15, 15, 55, 55]], dtype=np.float32),
        'scores': np.array([0.9, 0.85], dtype=np.float32),
        'class_ids': np.array([1, 1], dtype=np.int32)
    },
    1: {  # Frame 1
        'boxes': np.array([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=np.float32),
        'scores': np.array([0.88, 0.92], dtype=np.float32),
        'class_ids': np.array([1, 2], dtype=np.int32)
    },
    # ... more frames
}

# ✓ CORRECT: Process each frame independently
results = {}
for frame_id, detections in video_detections.items():
    keep_indices = rust_nms.multiclass_nms(
        detections['boxes'],
        detections['scores'],
        detections['class_ids'],
        iou_threshold=0.5,
        score_threshold=0.3
    )
    results[frame_id] = {
        'boxes': detections['boxes'][keep_indices],
        'scores': detections['scores'][keep_indices],
        'class_ids': detections['class_ids'][keep_indices]
    }

# ✗ WRONG: Don't concatenate across frames!
# This would allow frame 0 boxes to suppress frame 1 boxes!
all_boxes = np.vstack([d['boxes'] for d in video_detections.values()])
all_scores = np.concatenate([d['scores'] for d in video_detections.values()])
all_class_ids = np.concatenate([d['class_ids'] for d in video_detections.values()])
keep = rust_nms.multiclass_nms(all_boxes, all_scores, all_class_ids, 0.5)  # INCORRECT!
```

**Why?** A person standing at position (100, 100, 200, 200) in frame 1 should not suppress a different person at the same position in frame 50, even if they have similar scores. They are different detections in different temporal contexts.

## Use Cases

### Computer Vision
- **Object Detection**: Post-process YOLO/Faster R-CNN detections (per-frame for video)
- **Instance Segmentation**: Convert Mask R-CNN outputs to polygons
- **Semantic Segmentation**: Extract object boundaries from soft masks

### Robotics
- **Obstacle Detection**: Filter redundant bounding boxes in real-time streams
- **Scene Understanding**: Convert probability maps to geometric shapes

### Medical Imaging
- **Lesion Detection**: Remove overlapping candidate regions
- **Organ Segmentation**: Extract precise boundaries from probability maps
- **Video Endoscopy**: Process each frame independently for polyp detection

## References

See [AGENTS.md](AGENTS.md) for implementation references and algorithmic insights.

Key reference:
- [Modern-Cpp-NMS](https://github.com/developer0hye/Modern-Cpp-NMS) - C++ NMS implementation

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure `cargo test` and `python test_nms.py` pass
5. Submit a pull request

## Citation

If you use this in research, please cite:

```bibtex
@software{rust_nms,
  title = {rust-nms: Fast NMS and Mask-to-Polygon Conversion},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/GeorgePearse/rust-nms}
}
```

## Acknowledgments

- Built with [PyO3](https://github.com/PyO3/pyo3) and [maturin](https://github.com/PyO3/maturin)
- Inspired by [Modern-Cpp-NMS](https://github.com/developer0hye/Modern-Cpp-NMS)
- Created with Claude Code
