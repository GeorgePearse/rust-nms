# Agents & References

## ⚠️ CRITICAL: Pre-Commit Performance Requirements

**READ THIS FIRST BEFORE MAKING ANY CHANGES!**

This repository has **STRICT pre-commit hooks** that will **REJECT your commit** unless you meet both of these requirements:

### 1. All Tests Must Pass
```bash
cargo test --quiet
```

### 2. Performance Must Improve (NOT regress or stay the same)

The pre-commit hook runs benchmarks comparing your changes against `HEAD~1` (the previous commit). Your commit will be **REJECTED** if:
- ❌ Any benchmark shows a **regression** (slower performance)
- ❌ **No benchmarks show improvement** (same performance as before)

Your commit will be **ACCEPTED** only if:
- ✅ At least one benchmark shows **measurable improvement**
- ✅ No benchmarks show regressions
- ✅ All tests pass

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

### Benchmark Thresholds

The benchmark comparison script (`scripts/benchmark-compare.sh`) will:
1. Stash your changes
2. Run benchmarks on `HEAD~1` (previous commit)
3. Return to current state
4. Run benchmarks on your changes
5. Compare results and fail if no improvements detected

### Tips for Success

- **Think big**: Focus on algorithmic improvements, not micro-optimizations
- **Batch changes**: Combine multiple optimizations into a single commit
- **Profile first**: Understand what's actually slow before optimizing
- **Verify locally**: Always run `./scripts/benchmark-compare.sh` before committing

### Benchmark Locations

- Main benchmarks: `benches/nms_bench.rs`
- Baseline implementation: `src/lib.rs::nms_impl()`
- Alternative implementations: `src/nms_impls.rs`

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
