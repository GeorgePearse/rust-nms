# Agents & References

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

## Contributing

When adding new algorithms or implementations, please:
1. Document reference implementations in this file
2. Note key algorithmic insights or differences
3. Add to `/references/` directory if applicable
4. Update README with new capabilities
