use divan::{Bencher, black_box};
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rust_nms::{nms_impl, multiclass_nms_impl};
use rust_nms::nms_impls::nms_baseline;

fn main() {
    divan::main();
}

// Helper function to generate random boxes for benchmarking
fn generate_random_boxes(n: usize, seed: u64) -> (Array2<f32>, Array1<f32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    
    // Generate random boxes
    let mut boxes = Array2::zeros((n, 4));
    for i in 0..n {
        // Generate x1, y1 coordinates
        let x1 = rng.gen_range(0.0..900.0);
        let y1 = rng.gen_range(0.0..900.0);
        
        // Generate width and height
        let w = rng.gen_range(10.0..100.0);
        let h = rng.gen_range(10.0..100.0);
        
        // Set box coordinates [x1, y1, x2, y2]
        boxes[[i, 0]] = x1;
        boxes[[i, 1]] = y1;
        boxes[[i, 2]] = x1 + w;
        boxes[[i, 3]] = y1 + h;
    }
    
    // Generate random scores
    let scores = Array1::from_iter((0..n).map(|_| rng.gen_range(0.0..1.0)));
    
    (boxes, scores)
}

// Helper function to generate class IDs for multiclass benchmarks
fn generate_class_ids(n: usize, num_classes: usize, seed: u64) -> Array1<i32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    Array1::from_iter((0..n).map(|_| rng.gen_range(0..num_classes as i32)))
}

// Standard NMS benchmarks with different input sizes
#[divan::bench(
    consts = [
        100, // Small number of boxes
        1000, // Medium number of boxes
        10000, // Large number of boxes
    ],
)]
fn bench_nms<const N: usize>(bencher: Bencher) {
    let (boxes, scores) = generate_random_boxes(N, 42);
    
    bencher.bench(|| {
        let boxes_view = black_box(boxes.view());
        let scores_view = black_box(scores.view());
        nms_impl(boxes_view, scores_view, 0.5, None)
    })
}

// Benchmark NMS with max_detections
#[divan::bench(
    consts = [
        10000, // Large number of boxes
    ],
    args = [10, 100, 500], // Different max_detections values
)]
fn bench_nms_max_detections<const N: usize>(bencher: Bencher, max_dets: usize) {
    let (boxes, scores) = generate_random_boxes(N, 42);
    
    bencher.bench(|| {
        let boxes_view = black_box(boxes.view());
        let scores_view = black_box(scores.view());
        nms_impl(boxes_view, scores_view, 0.5, Some(max_dets))
    })
}

// Multiclass NMS benchmarks
#[divan::bench(
    consts = [
        10000, // Number of boxes
    ],
    args = [10, 50, 80], // Different numbers of classes (80 = COCO)
)]
fn bench_multiclass_nms<const N: usize>(bencher: Bencher, num_classes: usize) {
    let (boxes, scores) = generate_random_boxes(N, 42);
    let class_ids = generate_class_ids(N, num_classes, 43);
    
    bencher.bench(|| {
        let boxes_view = black_box(boxes.view());
        let scores_view = black_box(scores.view());
        let class_ids_view = black_box(class_ids.view());
        multiclass_nms_impl(
            boxes_view, 
            scores_view, 
            class_ids_view, 
            0.5, 
            None,
            None,
            None,
            None,
        )
    })
}

// Benchmark parallel multiclass NMS
#[divan::bench]
fn bench_multiclass_nms_parallel(bencher: Bencher) {
    const N: usize = 10000;
    const C: usize = 80;
    let (boxes, scores) = generate_random_boxes(N, 42);
    let class_ids = generate_class_ids(N, C, 43);
    
    bencher.bench(|| {
        let boxes_view = black_box(boxes.view());
        let scores_view = black_box(scores.view());
        let class_ids_view = black_box(class_ids.view());
        multiclass_nms_impl(
            boxes_view, 
            scores_view, 
            class_ids_view, 
            0.5, 
            None,
            None,
            None,
            Some(true),
        )
    })
}

// Benchmark baseline NMS (the one we're optimizing)
#[divan::bench(
    consts = [
        1000, // Small number of boxes
        10000, // Large number of boxes
    ],
)]
fn bench_nms_baseline<const N: usize>(bencher: Bencher) {
    let (boxes, scores) = generate_random_boxes(N, 42);
    
    bencher.bench(|| {
        let boxes_view = black_box(boxes.view());
        let scores_view = black_box(scores.view());
        nms_baseline(boxes_view, scores_view, 0.5, None)
    })
}

