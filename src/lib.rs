use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use ndarray::{Array2, ArrayView1, ArrayView2};

/// Bounding box representation
/// Performance is continuously tracked via automated benchmarks
#[derive(Debug, Clone, Copy)]
struct BBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    score: f32,
    index: usize,
}

impl BBox {
    fn area(&self) -> f32 {
        (self.x2 - self.x1).max(0.0) * (self.y2 - self.y1).max(0.0)
    }

    fn iou(&self, other: &BBox) -> f32 {
        let inter_x1 = self.x1.max(other.x1);
        let inter_y1 = self.y1.max(other.y1);
        let inter_x2 = self.x2.min(other.x2);
        let inter_y2 = self.y2.min(other.y2);

        let inter_area = (inter_x2 - inter_x1).max(0.0) * (inter_y2 - inter_y1).max(0.0);
        let union_area = self.area() + other.area() - inter_area;

        if union_area > 0.0 {
            inter_area / union_area
        } else {
            0.0
        }
    }
}

/// Fast Non-Maximum Suppression implementation
///
/// # Arguments
/// * `boxes` - Nx4 array of bounding boxes [x1, y1, x2, y2]
/// * `scores` - N-length array of confidence scores
/// * `iou_threshold` - IoU threshold for suppression (typically 0.5)
///
/// # Returns
/// * Array of indices of boxes to keep
fn nms_impl(boxes: ArrayView2<f32>, scores: ArrayView1<f32>, iou_threshold: f32) -> Vec<usize> {
    let n = boxes.nrows();

    if n == 0 {
        return Vec::new();
    }

    // Create bbox structs with scores
    let mut bboxes: Vec<BBox> = (0..n)
        .map(|i| BBox {
            x1: boxes[[i, 0]],
            y1: boxes[[i, 1]],
            x2: boxes[[i, 2]],
            y2: boxes[[i, 3]],
            score: scores[i],
            index: i,
        })
        .collect();

    // Sort by score descending
    bboxes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    let mut keep = Vec::new();
    let mut suppressed = vec![false; n];

    for i in 0..n {
        if suppressed[i] {
            continue;
        }

        keep.push(bboxes[i].index);

        for j in (i + 1)..n {
            if suppressed[j] {
                continue;
            }

            if bboxes[i].iou(&bboxes[j]) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    keep
}

/// Contour point for polygon extraction
#[derive(Debug, Clone, Copy, PartialEq)]
struct Point {
    x: i32,
    y: i32,
}

/// Moore-neighbor tracing for contour extraction
/// Extracts contours from a binary mask using the Moore-neighbor algorithm
fn trace_contour(mask: &Array2<bool>, start: Point) -> Vec<Point> {
    let (height, width) = mask.dim();
    let mut contour = Vec::new();

    // 8-connected neighbors in clockwise order starting from top
    let neighbors = [
        (-1, 0),  // N
        (-1, 1),  // NE
        (0, 1),   // E
        (1, 1),   // SE
        (1, 0),   // S
        (1, -1),  // SW
        (0, -1),  // W
        (-1, -1), // NW
    ];

    let mut current = start;
    let mut direction = 7; // Start looking from W

    loop {
        contour.push(current);

        let mut found = false;
        // Check 8 neighbors starting from the previous direction
        for i in 0..8 {
            let check_dir = (direction + i) % 8;
            let (dy, dx) = neighbors[check_dir];
            let ny = current.y + dy;
            let nx = current.x + dx;

            if ny >= 0 && ny < height as i32 && nx >= 0 && nx < width as i32 {
                if mask[[ny as usize, nx as usize]] {
                    current = Point { x: nx, y: ny };
                    direction = (check_dir + 5) % 8; // Look from 90 degrees CCW next time
                    found = true;
                    break;
                }
            }
        }

        if !found || (current == start && contour.len() > 1) {
            break;
        }

        // Safety check to prevent infinite loops
        if contour.len() > height * width {
            break;
        }
    }

    contour
}

/// Convert soft mask to segmentation polygons using marching squares-like approach
///
/// # Arguments
/// * `mask` - HxW array of scores from 0.0 to 1.0
/// * `threshold` - Threshold value for binarization (typically 0.5)
/// * `min_area` - Minimum polygon area to keep (in pixels)
///
/// # Returns
/// * Vector of polygons, where each polygon is a vector of (x, y) points
fn mask_to_polygons_impl(
    mask: ArrayView2<f32>,
    threshold: f32,
    min_area: usize,
) -> Vec<Vec<(f32, f32)>> {
    let (height, width) = mask.dim();

    // Binarize the mask
    let binary_mask: Array2<bool> = mask.map(|&v| v >= threshold);

    let mut visited = Array2::from_elem((height, width), false);
    let mut polygons = Vec::new();

    // Find all connected components
    for y in 0..height {
        for x in 0..width {
            if binary_mask[[y, x]] && !visited[[y, x]] {
                // Found a new component, trace its contour
                let start = Point { x: x as i32, y: y as i32 };
                let contour = trace_contour(&binary_mask, start);

                // Mark visited (flood fill the component)
                let mut stack = vec![(y, x)];
                while let Some((cy, cx)) = stack.pop() {
                    if visited[[cy, cx]] {
                        continue;
                    }
                    visited[[cy, cx]] = true;

                    // Add neighbors to stack
                    for (dy, dx) in [(-1, 0), (1, 0), (0, -1), (0, 1)].iter() {
                        let ny = cy as i32 + dy;
                        let nx = cx as i32 + dx;
                        if ny >= 0 && ny < height as i32 && nx >= 0 && nx < width as i32 {
                            let ny = ny as usize;
                            let nx = nx as usize;
                            if binary_mask[[ny, nx]] && !visited[[ny, nx]] {
                                stack.push((ny, nx));
                            }
                        }
                    }
                }

                // Convert contour to polygon and filter by area
                if contour.len() >= min_area {
                    let polygon: Vec<(f32, f32)> = contour
                        .iter()
                        .map(|p| (p.x as f32, p.y as f32))
                        .collect();

                    if !polygon.is_empty() {
                        polygons.push(polygon);
                    }
                }
            }
        }
    }

    polygons
}

/// Python wrapper for NMS
#[pyfunction]
#[pyo3(signature = (boxes, scores, iou_threshold=0.5))]
fn nms<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<f32>,
    scores: PyReadonlyArray1<f32>,
    iou_threshold: f32,
) -> PyResult<&'py PyArray1<usize>> {
    let boxes_view = boxes.as_array();
    let scores_view = scores.as_array();

    if boxes_view.ncols() != 4 {
        return Err(PyValueError::new_err("boxes must have shape (N, 4)"));
    }

    if boxes_view.nrows() != scores_view.len() {
        return Err(PyValueError::new_err("boxes and scores must have same length"));
    }

    let keep = nms_impl(boxes_view, scores_view, iou_threshold);
    Ok(PyArray1::from_vec(py, keep))
}

/// Python wrapper for mask to polygons conversion
#[pyfunction]
#[pyo3(signature = (mask, threshold=0.5, min_area=10))]
fn mask_to_polygons(
    mask: PyReadonlyArray2<f32>,
    threshold: f32,
    min_area: usize,
) -> PyResult<Vec<Vec<(f32, f32)>>> {
    let mask_view = mask.as_array();

    if threshold < 0.0 || threshold > 1.0 {
        return Err(PyValueError::new_err("threshold must be between 0 and 1"));
    }

    let polygons = mask_to_polygons_impl(mask_view, threshold, min_area);
    Ok(polygons)
}

/// Python module definition
#[pymodule]
fn rust_nms(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(nms, m)?)?;
    m.add_function(wrap_pyfunction!(mask_to_polygons, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_bbox_area() {
        let bbox = BBox {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
            score: 0.9,
            index: 0,
        };
        assert_eq!(bbox.area(), 100.0);
    }

    #[test]
    fn test_bbox_iou() {
        let bbox1 = BBox {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
            score: 0.9,
            index: 0,
        };
        let bbox2 = BBox {
            x1: 5.0,
            y1: 5.0,
            x2: 15.0,
            y2: 15.0,
            score: 0.8,
            index: 1,
        };
        let iou = bbox1.iou(&bbox2);
        assert!((iou - 0.142857).abs() < 0.001); // 25 / 175
    }

    #[test]
    fn test_nms_basic() {
        let boxes = array![
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [50.0, 50.0, 60.0, 60.0],
        ];
        let scores = array![0.9, 0.8, 0.95];

        let keep = nms_impl(boxes.view(), scores.view(), 0.5);
        assert_eq!(keep.len(), 2);
        assert!(keep.contains(&0) || keep.contains(&1));
        assert!(keep.contains(&2));
    }

    #[test]
    fn test_mask_to_polygons_simple() {
        let mask = array![
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ];

        let polygons = mask_to_polygons_impl(mask.view(), 0.5, 1);
        assert!(!polygons.is_empty());
    }
}
