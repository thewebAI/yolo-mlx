# Changelog

## 0.2.0 — 2026-04-01

- **Tracking**: Batched Kalman filter updates (single `mx.linalg.inv` call per association stage)
- **Tracking**: Batch-precomputed coordinates reduce MLX graph dispatch overhead
- **Tracking**: MLX tracking now matches or exceeds PyTorch MPS speed
- **Packaging**: Moved `scipy` from core dependencies to `[tracking]` extra
- **Packaging**: Added clear error message when tracking dependencies are missing
- **Benchmarks**: Added tracking benchmark scripts and charts

## 0.1.0 — 2026-03-18

- Initial release: detection, training, COCO validation
