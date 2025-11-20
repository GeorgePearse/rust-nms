# Agents & References

## ⚠️ CRITICAL: Pre-Commit Performance Requirements

**READ THIS FIRST BEFORE MAKING ANY CHANGES!**

This repository has **STRICT pre-commit hooks** that will **REJECT your commit** unless you meet both of these requirements:

> **🚨 IMPORTANT:** Commits are AUTOMATICALLY REJECTED if benchmarks don't show performance improvements. The pre-commit hook compares your changes against the previous commit and blocks any commit that doesn't make the code measurably faster.

### 1. All Tests Must Pass
```bash
cargo test --quiet
```

### 2. Benchmarks MUST Show Performance Improvement

**CRITICAL REQUIREMENT:** The pre-commit hook runs benchmarks comparing your changes against `HEAD~1` (the previous commit).

**Your commit will be REJECTED if:**
- ❌ Any benchmark shows a **regression** (slower performance)
- ❌ **No benchmarks show improvement** (same performance as before)
- ❌ Benchmarks show identical performance to the previous commit

**Your commit will be ACCEPTED only if:**
- ✅ **At least one benchmark shows measurable improvement** (faster execution)
- ✅ **No benchmarks show regressions** (none can be slower)
- ✅ All tests pass

**This means:** Every commit MUST make the code measurably faster. The benchmark comparison script will automatically run during `git commit` and will block commits that don't improve performance. There is no way to commit code that doesn't show benchmark improvements except by using `git commit --no-verify` (not recommended).

### What This Means for Agents

**DO NOT attempt to commit:**
- Minor refactoring that doesn't improve performance
- Documentation-only changes
- Small code style improvements
- Tiny optimizations that won't show measurable gains

**ONLY commit when you have:**
- Significant algorithmic improvements (e.g., spatial indexing, vectorization)
- Cache optimization strategies (e.g., SoA vs AoS layout)
- Memory access pattern improvements
- Branch prediction optimizations
- Early rejection strategies that skip expensive calculations

### How to Test Before Committing

```bash
# Run the pre-commit hook manually
./scripts/pre-commit

# Or test benchmarks directly
./scripts/benchmark-compare.sh
```

### How Benchmark Validation Works

The benchmark comparison script (`scripts/benchmark-compare.sh`) automatically runs during `git commit` and performs these steps:

1. **Stash your uncommitted changes** (if any)
2. **Checkout HEAD~1** and run benchmarks on the previous commit
3. **Return to your current changes** and run benchmarks
4. **Compare the results:**
   - If any benchmark is slower → **REJECT commit**
   - If no benchmark is faster → **REJECT commit**
   - If at least one benchmark is faster and none are slower → **ACCEPT commit**
5. **Restore your working state**

**Special cases:**
- If HEAD~1 has no benchmarks (e.g., first commit with benchmark infrastructure), the check is skipped
- The script compares execution times and looks for "faster" or "slower" indicators in benchmark output

### Tips for Success

- **Think big**: Focus on algorithmic improvements, not micro-optimizations
- **Batch changes**: Combine multiple optimizations into a single commit
- **Profile first**: Understand what's actually slow before optimizing
- **Verify locally**: Always run `./scripts/benchmark-compare.sh` before committing

### Benchmark Locations

- Main benchmarks: `benches/nms_bench.rs`
- **Primary/Best implementation**: `src/lib.rs::nms_impl()` ⭐
- Alternative implementations (research): `src/nms_impls.rs`

### Implementation Philosophy

**🔬 All implementations are kept for research purposes:**

This repository maintains multiple NMS implementations in `src/nms_impls.rs` even if they are not the fastest. These alternative implementations serve important research purposes:

- **Baseline comparisons**: Older implementations provide benchmarking baselines to measure improvement
- **Educational value**: Different optimization strategies demonstrate various performance techniques
- **Experimentation**: Alternative approaches may perform better under different workloads or hardware
- **Documentation**: Code serves as documentation of what was tried and why certain approaches work better

**⚠️ Do NOT delete old implementations** - they are intentionally preserved for historical and research purposes.

**✨ Current Best Implementation:**

The primary, production-ready NMS implementation is **`nms_impl()` in `src/lib.rs`**, which currently includes:

- **Structure of Arrays (SoA) memory layout** for cache-friendly coordinate access
- **Spatial indexing with grid-based filtering** (activated for n > 500 boxes)
  - Reduces IoU comparisons from O(n²) to O(n·k) where k is local density
  - Uses adaptive cell padding and fallback heuristics
- **Immediate suppression marking** to eliminate allocation overhead and enable early exits
- **Manual loop unrolling** (blocks of 4) for instruction-level parallelism
- **Unsafe operations** for bounds check elimination in hot paths
- **Early rejection tests** before expensive IoU calculations
- **Adaptive capacity estimation** based on input size

When adding new optimizations, always benchmark against `nms_impl()` and update this pointer if you create a faster implementation.

---

## NMS Optimization Research & Evaluation

### Research Sources (November 2024)

Extensive research was conducted to find cutting-edge NMS optimizations:

**Academic Papers Reviewed:**
1. **"Accelerating Non-Maximum Suppression: A Graph Theory Perspective"** (2024, arXiv:2409.20520)
   - QSI-NMS: Divide-and-conquer achieving O(n log n) with 6.2×-10.7× speedup
   - BOE-NMS: Locality-focused achieving 5.1× speedup with no mAP loss
   - Reveals NMS as a graph theory problem with intrinsic structure

2. **Hardware Accelerator Studies:**
   - Sort-less NMS approaches (removes pre-sorting bottleneck)
   - Position-based bit tables for data reusing
   - Multi-thread computing and binary max engines

3. **Industrial Implementations:**
   - PyTorch torchvision NMS (baseline reference)
   - NVIDIA TensorRT efficientNMS
   - Modern-Cpp-NMS (score zeroing technique)

### Optimization Techniques Evaluated

#### ✅ **Implemented & Successful:**

1. **Spatial Indexing with Grid-Based Filtering** (60% speedup on 10K boxes)
   - Reduces comparisons from O(n²) to O(n·k) where k is local density
   - Adaptive cell padding and fallback heuristics
   - Only activates for n > 500 to avoid overhead on small inputs

2. **Structure of Arrays (SoA) Memory Layout**
   - Separates x1, y1, x2, y2 into contiguous arrays
   - Dramatically improves cache locality
   - Enables better vectorization opportunities

3. **Immediate Suppression Marking**
   - Eliminates batch collection overhead
   - Enables early exits within same iteration
   - Reduces memory allocations

4. **Manual Loop Unrolling** (blocks of 4)
   - Explicit unrolling for better instruction-level parallelism
   - Prefetches suppression status
   - Better CPU pipeline utilization

5. **Early Rejection via Bounding Box Test**
   - Cheap axis-aligned overlap check before IoU
   - Eliminates ~30-40% of IoU calculations

#### ❌ **Tested But Not Beneficial:**

1. **Area-Ratio Early Rejection**
   - Theory: Skip pairs where min(area_i, area_j) / max(area_i, area_j) ≤ threshold
   - Result: ~7% **slower** (670µs vs 628µs for 1000 boxes)
   - Reason: Division operation overhead exceeds benefit; most overlapping boxes have similar areas

2. **Division-Free IoU Comparison**
   - Theory: Rearrange `inter/(union) > t` to `inter*(1+t) > t*(area_i+area_j)` to replace division with multiplications
   - Result: ~3.4× **slower** (1022µs vs 628µs for 1000 boxes)
   - Reason: Modern FPUs pipeline division efficiently; extra multiplications add latency

3. **Pre-computing Area Reciprocals**
   - Theory: Store 1/area to replace division with multiplication
   - Result: Not implemented - union area computed dynamically, can't pre-compute reciprocal

#### 🔬 **Identified But Not Implemented (Require Algorithm Changes):**

1. **QSI-NMS / BOE-NMS** (Graph Theory Approaches)
   - Would require fundamental restructuring of algorithm
   - Promises 5×-10× speedup but changes output ordering
   - Potential future research direction

2. **Sort-Less NMS**
   - Removes pre-sorting step
   - Requires different data structures (binary max engine)
   - Hardware-oriented optimization

3. **Diagonal Distance Pre-filtering**
   - Check box center distances before IoU
   - Requires square root or squared distance calculation
   - Unclear if cheaper than current bounding box test

### Key Insights from Research

1. **CPU vs GPU optimizations differ**: Many fast NMS papers focus on GPU/hardware accelerators. CPU optimizations must respect cache hierarchy and branch prediction.

2. **Division isn't always expensive**: Modern CPUs have pipelined FPUs where one division may be cheaper than multiple multiplications + comparisons.

3. **Micro-optimizations have limits**: After SoA layout + spatial indexing + early rejection, further gains require algorithmic changes.

4. **Spatial locality matters most**: The 60% speedup from spatial indexing dwarfs other optimizations. Exploiting spatial structure is key.

5. **Real-world box distributions**: Object detection produces boxes with high overlap and similar areas, making area-ratio filtering ineffective.

### References

- Modern-Cpp-NMS: https://github.com/developer0hye/Modern-Cpp-NMS
- PyTorch Vision NMS: https://github.com/pytorch/vision (torchvision/csrc/ops/cpu/nms_kernel.cpp)
- Graph Theory NMS Paper: https://arxiv.org/abs/2409.20520
- FastNMS: https://github.com/gdroguski/FastNMS

---

## Implementation References

### Non-Maximum Suppression (NMS)

The NMS implementation in this project was informed by the following reference:

- **Modern C++ NMS Implementation**
  - Repository: [developer0hye/Modern-Cpp-NMS](https://github.com/developer0hye/Modern-Cpp-NMS)
  - Location: `/references/Modern-Cpp-NMS/`
  - Key insights:
    - Efficient in-place modification approach
    - Multi-class NMS with class grouping via sorting
    - Score zeroing technique for suppression tracking
    - Clean IoU calculation with epsilon for numerical stability

Our Rust implementation differs in several ways:
- Returns indices instead of modifying input (more functional approach)
- Separate tracking of suppressed boxes for clarity
- Built for Python interop via PyO3/maturin
- Zero-copy array views using ndarray

### Mask to Polygon Conversion

The soft mask to polygon conversion uses:
- **Moore-neighbor tracing** for contour extraction
- **Flood-fill** for connected component analysis
- Binarization with configurable threshold
- Minimum area filtering to remove noise

This algorithm is inspired by classical computer vision techniques:
- Moore boundary tracing (1968)
- Marching squares algorithm concepts
- Connected component labeling

## AI Agents Used

This project was created with assistance from:
- **Claude Code** (Anthropic) - Architecture design, Rust implementation, Python bindings
- **GitHub Copilot** - Code completion suggestions

## Development Tools

- **Rust**: Core implementation language
- **PyO3**: Python bindings
- **maturin**: Build system for Rust/Python integration
- **ndarray**: Rust numerical computing
- **numpy**: Python array interface

## Implementation Constraints

### Design Decisions

**Per-Frame Processing (No Built-in Frame Isolation):**

This library intentionally does NOT include frame_id or image_id parameters. NMS functions operate on a **single batch** of boxes. This is a deliberate design choice:

- **Rationale**: Frame batching is application-specific and should be handled by the caller
- **Performance**: Adding frame grouping would add overhead for single-image use cases
- **Flexibility**: Users have full control over batching strategy
- **Correctness**: Forces explicit per-frame processing, preventing accidental cross-frame suppression

**Critical Rule:** Boxes from different frames/images should NEVER be processed together. A box in frame 1 should not suppress a box in frame 50, even with identical coordinates, as they represent different temporal contexts.

**Correct Usage Pattern:**
```python
# ✓ CORRECT: Process each frame independently
for frame_id, frame_boxes, frame_scores in video_stream:
    keep = rust_nms.nms(frame_boxes, frame_scores, iou_threshold=0.5)

# ✗ WRONG: Never concatenate boxes from different frames
all_boxes = np.vstack(all_frame_boxes)  # DON'T DO THIS!
keep = rust_nms.nms(all_boxes, all_scores, 0.5)  # INCORRECT!
```

### Algorithms Not Supported

This project explicitly does not support the following algorithms:

- **Soft-NMS**: We have chosen not to implement Soft-NMS due to:
  - Increased computational complexity
  - Limited practical benefits in most real-world applications
  - Maintenance overhead for additional code paths
  - Preference for simplicity and performance of standard NMS

Any pull requests adding Soft-NMS functionality will be declined. The project focuses on high-performance implementations of standard NMS and mask-to-polygon conversion.

## Contributing

When adding new algorithms or implementations, please:
1. Document reference implementations in this file
2. Note key algorithmic insights or differences
3. Add to `/references/` directory if applicable
4. Update README with new capabilities
5. Respect the implementation constraints listed above
