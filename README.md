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

### `nms(boxes, scores, iou_threshold=0.5)`

Non-Maximum Suppression for bounding boxes.

**Parameters:**
- `boxes`: `np.ndarray[float32]` - Shape (N, 4), format [x1, y1, x2, y2]
- `scores`: `np.ndarray[float32]` - Shape (N,), confidence scores
- `iou_threshold`: `float` - IoU threshold for suppression (default: 0.5)

**Returns:**
- `np.ndarray[uint]` - Indices of boxes to keep

**Algorithm:**
1. Sort boxes by score (descending)
2. Iteratively select highest scoring box
3. Suppress overlapping boxes above IoU threshold
4. Return indices of non-suppressed boxes

**Time Complexity:** O(N²) worst case, typically much faster

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

## Use Cases

### Computer Vision
- **Object Detection**: Post-process YOLO/Faster R-CNN detections
- **Instance Segmentation**: Convert Mask R-CNN outputs to polygons
- **Semantic Segmentation**: Extract object boundaries from soft masks

### Robotics
- **Obstacle Detection**: Filter redundant bounding boxes
- **Scene Understanding**: Convert probability maps to geometric shapes

### Medical Imaging
- **Lesion Detection**: Remove overlapping candidate regions
- **Organ Segmentation**: Extract precise boundaries from probability maps

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
