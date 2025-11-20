use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use ndarray::{ArrayView1, ArrayView2};

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

    // Pre-calculate areas to avoid re-computation in the loop
    // Area = (x2 - x1) * (y2 - y1)
    let areas: Vec<f32> = (0..n)
        .map(|i| {
            let w = (boxes[[i, 2]] - boxes[[i, 0]]).max(0.0);
            let h = (boxes[[i, 3]] - boxes[[i, 1]]).max(0.0);
            w * h
        })
        .collect();

    // Sort indices by score descending
    // We use a vector of indices to avoid copying the bbox data
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = Vec::with_capacity(n / 10); // Heuristic pre-allocation
    let mut suppressed = vec![false; n];

    for i in 0..n {
        let idx_i = order[i];
        if suppressed[idx_i] {
            continue;
        }

        keep.push(idx_i);
        
        let x1_i = boxes[[idx_i, 0]];
        let y1_i = boxes[[idx_i, 1]];
        let x2_i = boxes[[idx_i, 2]];
        let y2_i = boxes[[idx_i, 3]];
        let area_i = areas[idx_i];

        // Check against all subsequent (lower score) boxes
        // Note: iterating j > i in the SORTED order
        // Optimization: Iterate directly over the slice of remaining indices
        for &idx_j in &order[(i + 1)..] {
            if suppressed[idx_j] {
                continue;
            }

            // Calculate IoU
            let inter_x1 = x1_i.max(boxes[[idx_j, 0]]);
            let inter_y1 = y1_i.max(boxes[[idx_j, 1]]);
            let inter_x2 = x2_i.min(boxes[[idx_j, 2]]);
            let inter_y2 = y2_i.min(boxes[[idx_j, 3]]);

            let w = (inter_x2 - inter_x1).max(0.0);
            let h = (inter_y2 - inter_y1).max(0.0);
            let inter_area = w * h;

            if inter_area > 0.0 {
                let union_area = area_i + areas[idx_j] - inter_area;
                if union_area > 0.0 && (inter_area / union_area) > iou_threshold {
                    suppressed[idx_j] = true;
                }
            }
        }
    }

    keep
}

enum SoftNmsMethod {
    Linear,
    Gaussian,
}

fn soft_nms_impl(
    boxes: ArrayView2<f32>,
    scores: ArrayView1<f32>,
    method: SoftNmsMethod,
    sigma: f32,
    iou_threshold: f32,
    score_threshold: f32,
) -> (Vec<usize>, Vec<f32>) {
    let n = boxes.nrows();
    let mut updated_scores = scores.to_vec();
    let mut indices: Vec<usize> = (0..n).collect();
    let mut keep = Vec::with_capacity(n);

    // Pre-calculate areas
    let areas: Vec<f32> = (0..n)
        .map(|i| {
            let w = (boxes[[i, 2]] - boxes[[i, 0]]).max(0.0);
            let h = (boxes[[i, 3]] - boxes[[i, 1]]).max(0.0);
            w * h
        })
        .collect();

    for i in 0..n {
        // Find max score among remaining indices (i..n)
        let mut max_score = -f32::INFINITY;
        let mut max_pos = i;

        for (pos, &idx) in indices.iter().enumerate().skip(i) {
            let s = updated_scores[idx];
            if s > max_score {
                max_score = s;
                max_pos = pos;
            }
        }

        // Swap the best box to position i
        indices.swap(i, max_pos);
        let current_idx = indices[i];
        let current_score = updated_scores[current_idx];

        if current_score < score_threshold {
            break;
        }

        keep.push(current_idx);

        let x1 = boxes[[current_idx, 0]];
        let y1 = boxes[[current_idx, 1]];
        let x2 = boxes[[current_idx, 2]];
        let y2 = boxes[[current_idx, 3]];
        let area = areas[current_idx];

        // Update scores of remaining boxes
        for &idx in &indices[(i + 1)..] {
            let inter_x1 = x1.max(boxes[[idx, 0]]);
            let inter_y1 = y1.max(boxes[[idx, 1]]);
            let inter_x2 = x2.min(boxes[[idx, 2]]);
            let inter_y2 = y2.min(boxes[[idx, 3]]);

            let w = (inter_x2 - inter_x1).max(0.0);
            let h = (inter_y2 - inter_y1).max(0.0);
            let inter_area = w * h;

            let mut iou = 0.0;
            if inter_area > 0.0 {
                let union = area + areas[idx] - inter_area;
                if union > 0.0 {
                    iou = inter_area / union;
                }
            }

            let weight = match method {
                SoftNmsMethod::Linear => {
                    if iou > iou_threshold {
                        1.0 - iou
                    } else {
                        1.0
                    }
                }
                SoftNmsMethod::Gaussian => (-iou * iou / sigma).exp(),
            };

            updated_scores[idx] *= weight;
        }
    }
    
    let kept_scores = keep.iter().map(|&idx| updated_scores[idx]).collect();
    (keep, kept_scores)
}

/// Contour point for polygon extraction
#[derive(Debug, Clone, Copy, PartialEq)]
struct Point {
    x: i32,
    y: i32,
}

/// Moore-neighbor tracing for contour extraction
/// Extracts contours from a binary source using the Moore-neighbor algorithm
fn trace_contour(
    mask: ArrayView2<f32>,
    threshold: f32,
    start: Point,
) -> Vec<Point> {
    let (height, width) = mask.dim();
    let h_i32 = height as i32;
    let w_i32 = width as i32;
    
    let mut contour = Vec::with_capacity(100);

    // 8-connected neighbors in clockwise order starting from top (N)
    // (dy, dx)
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
    let mut direction = 7; // Start looking from West (since we entered from left/top)

    // Safety limit to prevent infinite loops in pathological cases
    let max_steps = width * height;

    loop {
        contour.push(current);

        let mut found = false;
        // Check 8 neighbors in clockwise order
        for i in 0..8 {
            let check_dir = (direction + i) % 8;
            let (dy, dx) = neighbors[check_dir];
            let ny = current.y + dy;
            let nx = current.x + dx;

            // Bounds check and threshold check
            if ny >= 0 && ny < h_i32 && nx >= 0 && nx < w_i32 && mask[[ny as usize, nx as usize]] >= threshold {
                current = Point { x: nx, y: ny };
                direction = (check_dir + 5) % 8; 
                found = true;
                break;
            }
        }

        if !found || (current == start && contour.len() > 1) {
            break;
        }

        if contour.len() >= max_steps {
            break;
        }
    }

    contour
}

/// Convert soft mask to segmentation polygons
fn mask_to_polygons_impl(
    mask: ArrayView2<f32>,
    threshold: f32,
    min_area: usize,
) -> Vec<Vec<(f32, f32)>> {
    let (height, width) = mask.dim();
    let h_i32 = height as i32;
    let w_i32 = width as i32;

    // Use a 1D boolean vector for visited status
    let mut visited = vec![false; height * width];
    let mut polygons = Vec::new();
    
    // Re-use stack to avoid allocation in loop
    let mut stack = Vec::with_capacity(1024);

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            
            // Check if pixel is solid and not visited
            // We perform the check here to avoid function call overhead if not needed
            if !visited[idx] && mask[[y, x]] >= threshold {
                // Found a new component
                
                // 1. Trace contour
                let start = Point { x: x as i32, y: y as i32 };
                let contour = trace_contour(mask, threshold, start);
                
                // 2. Flood fill to mark this entire component as visited
                // Optimization: Mark visited WHEN PUSHING to stack to avoid duplicates
                stack.clear();
                stack.push((y, x));
                visited[idx] = true;

                while let Some((cy, cx)) = stack.pop() {
                    // Check 4 neighbors (sufficient for connectivity)
                    // Inline neighbors
                    let neighbors = [
                        (cy as i32 - 1, cx as i32),
                        (cy as i32 + 1, cx as i32),
                        (cy as i32, cx as i32 - 1),
                        (cy as i32, cx as i32 + 1)
                    ];

                    for &(ny_i, nx_i) in &neighbors {
                        if ny_i >= 0 && ny_i < h_i32 && nx_i >= 0 && nx_i < w_i32 {
                            let ny = ny_i as usize;
                            let nx = nx_i as usize;
                            let nidx = ny * width + nx;
                            
                            if !visited[nidx] && mask[[ny, nx]] >= threshold {
                                visited[nidx] = true; // Mark immediately!
                                stack.push((ny, nx));
                            }
                        }
                    }
                }

                // 3. Convert to polygon and filter
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

/// Python wrapper for Soft-NMS
#[pyfunction]
#[pyo3(signature = (boxes, scores, method="linear", sigma=0.5, iou_threshold=0.3, score_threshold=0.001))]
fn soft_nms<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<f32>,
    scores: PyReadonlyArray1<f32>,
    method: &str,
    sigma: f32,
    iou_threshold: f32,
    score_threshold: f32,
) -> PyResult<(&'py PyArray1<usize>, &'py PyArray1<f32>)> {
    let boxes_view = boxes.as_array();
    let scores_view = scores.as_array();

    if boxes_view.ncols() != 4 {
        return Err(PyValueError::new_err("boxes must have shape (N, 4)"));
    }
    if boxes_view.nrows() != scores_view.len() {
        return Err(PyValueError::new_err("boxes and scores must have same length"));
    }

    let method_enum = match method {
        "linear" => SoftNmsMethod::Linear,
        "gaussian" => SoftNmsMethod::Gaussian,
        _ => return Err(PyValueError::new_err("method must be 'linear' or 'gaussian'")),
    };

    let (indices, new_scores) = soft_nms_impl(
        boxes_view,
        scores_view,
        method_enum,
        sigma,
        iou_threshold,
        score_threshold,
    );

    Ok((
        PyArray1::from_vec(py, indices),
        PyArray1::from_vec(py, new_scores),
    ))
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

    if !(0.0..=1.0).contains(&threshold) {
        return Err(PyValueError::new_err("threshold must be between 0 and 1"));
    }

    let polygons = mask_to_polygons_impl(mask_view, threshold, min_area);
    Ok(polygons)
}

/// Python module definition
#[pymodule]
fn rust_nms(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(nms, m)?)?;
    m.add_function(wrap_pyfunction!(soft_nms, m)?)?;
    m.add_function(wrap_pyfunction!(mask_to_polygons, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_nms_basic() {
        let boxes = array![
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [50.0, 50.0, 60.0, 60.0],
        ];
        let scores = array![0.9, 0.8, 0.95];

        let keep = nms_impl(boxes.view(), scores.view(), 0.5);
        
        // Should keep index 2 (score 0.95) and index 0 (score 0.9)
        // Index 1 (score 0.8) overlaps with index 0 (IoU > 0.5) so it should be suppressed
        assert!(keep.contains(&2));
        assert!(keep.contains(&0));
        assert!(!keep.contains(&1));
    }
    
    #[test]
    fn test_nms_empty() {
        let boxes = Array2::<f32>::zeros((0, 4));
        let scores = ArrayView1::<f32>::from(&[]);
        let keep = nms_impl(boxes.view(), scores, 0.5);
        assert!(keep.is_empty());
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
        assert_eq!(polygons.len(), 1);
        // Square has 4 corners + duplicate start point sometimes depending on trace
        // Trace: (1,1) -> (1,2) -> (2,2) -> (2,1) -> (1,1)
        assert!(polygons[0].len() >= 4);
    }
    
    #[test]
    fn test_mask_complex_shape() {
        // C-shape
        let mask = array![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let polygons = mask_to_polygons_impl(mask.view(), 0.5, 1);
        assert_eq!(polygons.len(), 1);
    }

    #[test]
    fn test_soft_nms_linear() {
        let boxes = array![
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0], // Exact overlap
        ];
        let scores = array![0.9, 0.8];
        
        // With linear soft NMS, the second box (score 0.8) has IoU 1.0 with the first.
        // Weight = 1.0 - 1.0 = 0.0
        // New score = 0.8 * 0.0 = 0.0
        // If score_threshold is 0.001, it should be dropped
        
        let (indices, new_scores) = soft_nms_impl(
            boxes.view(), 
            scores.view(), 
            SoftNmsMethod::Linear, 
            0.5, 
            0.3, 
            0.001
        );
        
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
        assert!((new_scores[0] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_soft_nms_gaussian() {
        let boxes = array![
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0], // High overlap
        ];
        let scores = array![0.9, 0.8];
        
        // Gaussian decay
        // IoU ~ 0.68
        // Weight = exp(-0.68^2 / 0.5) = exp(-0.4624 / 0.5) = exp(-0.9248) ~= 0.396
        // New score = 0.8 * 0.396 ~= 0.317
        // Should keep both if threshold is low
        
        let (indices, new_scores) = soft_nms_impl(
            boxes.view(), 
            scores.view(), 
            SoftNmsMethod::Gaussian, 
            0.5, 
            0.3, 
            0.1
        );
        
        assert_eq!(indices.len(), 2);
        assert_eq!(indices[0], 0); // First one is highest score
        assert_eq!(indices[1], 1);
        assert!(new_scores[1] < 0.8); // Score should decay
    }
}
