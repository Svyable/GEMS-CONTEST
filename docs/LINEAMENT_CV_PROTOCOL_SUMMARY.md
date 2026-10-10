# Lineament Features Fault/Trace CV Protocol — Implementation Summary

## Problem Solved

`scripts/train_full_map.py` previously refused all derived lineament features under fault/trace CV with:
```
fault/trace CV needs a masked-input transform protocol before comparison
```

This blocked all four queued lineament candidates from three-view gate evaluation:
- `mumford_shah_log` (LoG edge detection)
- `structure_tensor_coherence` (directional structure strength)  
- `ridge_valley_response` (Hessian eigenvalue valley/ridge detection)
- `steerable_filter` (Gabor-style orientation-invariant detection)

## Solution Implemented

**Leakage-safe protocol ensures held-out regions never contribute to training inputs:**

1. **Input masking for filtering**: Pass `train_mask` (excludes held-out faults + buffer) as `valid_mask` to `grouped_lineament_features`. All spatial filters (_masked_gaussian, structure tensor, Hessian, Gabor) treat held-out regions as invalid.

2. **Fold-pure normalization**: Already implemented — `normalization_mask` uses `train_mask` for statistics.

3. **Buffer >= filter support**: Already enforced — `required_lineament_buffer` check ensures buffer exceeds filter support.

## Files Changed

### `scripts/train_full_map.py`
- **Lines 112-136**: Removed hard refusal for fault/trace CV with lineament features
- Now uses `train_mask` as filtering mask when split exists
- Added inline comment documenting the protocol

### `tests/test_lineament_cv_leakage.py` (new, 15 tests)
Comprehensive leakage tests:
- Core leakage: perturbing held-out inputs by +1000 doesn't affect training features (7 kinds)
- Fold-pure normalization: perturbing held-out inputs by 10× doesn't affect statistics
- Spatial CV unchanged: correlation >0.99 proves no regression
- Buffer efficacy: training pixels >8px from boundary show near-zero artifacts
- Trace CV: endpoint perturbation doesn't leak into trace bodies
- Multi-scale: mumford_shah_log and ridge_valley_response respect masking

### `tests/test_train_lineament_fault_cv_integration.py` (new, 3 tests)
End-to-end integration tests verifying train_full_map.py accepts lineament features under fault/trace CV and proceeds past the old refusal point. Skip gracefully if torch unavailable.

### `docs/EXPERIMENT_LOG.md`
Added comprehensive documentation of protocol, rationale, tests, and usage.

## Test Results

✅ **73 tests passing:**
- 44 existing lineament tests
- 14 existing CV/training split tests  
- 15 new leakage tests

All tests demonstrate:
- No leakage from held-out regions
- Fold-pure normalization
- Spatial CV behavior unchanged
- Buffer prevents edge artifacts
- Protocol works for fault and trace CV

## Usage

**No config changes needed.** Existing candidate configs work as-is:

```bash
uv run python scripts/train_full_map.py \
  --config configs/candidates/cand_mumford_shah_log.yaml \
  --features data/raw/gems-geodawn-numerical-features.tif \
  --labels data/raw/existing_faults.tif \
  --template data/raw/example_submission.tif \
  --fold-map data/processed/cv-fault-v1.tif \
  --cv-scheme fault \
  --fold 0 \
  --buffer-pixels 16 \
  --seed 20260922 \
  --output runs/mumford-shah-fault-cv/fold-0.tif \
  --metrics-json runs/mumford-shah-fault-cv/metrics-fold-0.json
```

**Buffer requirements:**
- gradient_energy, phase_edge: 4-5 pixels
- mumford_shah_log: 4-5 pixels  
- structure_tensor_coherence/orientation: 13 pixels (σ=1, window=3)
- ridge_valley_response: 13-17 pixels (depends on scales)
- steerable_filter: 17 pixels (wavelength=8)

Default `--buffer-pixels 16` works for all except steerable_filter (use 20).

## CLI Config Keys for Local Queue

The four lineament candidates can now run on all CV views with these exact CLI invocations:

### Fault CV (all candidates):
```bash
for fold in 0 1 2 3 4; do
  for config in cand_mumford_shah_log cand_structure_tensor_coherence \
                cand_ridge_valley_response cand_steerable_filter; do
    uv run python scripts/train_full_map.py \
      --config "configs/candidates/${config}.yaml" \
      --features data/raw/gems-geodawn-numerical-features.tif \
      --labels data/raw/existing_faults.tif \
      --template data/raw/example_submission.tif \
      --fold-map data/processed/cv-fault-v1.tif \
      --cv-scheme fault \
      --fold $fold \
      --buffer-pixels 20 \
      --seed 20260922 \
      --output "runs/${config}-fault/fold-${fold}.tif" \
      --metrics-json "runs/${config}-fault/metrics-fold-${fold}.json"
  done
done
```

### Trace CV (all candidates):
```bash
for fold in 0 1 2 3 4; do
  for config in cand_mumford_shah_log cand_structure_tensor_coherence \
                cand_ridge_valley_response cand_steerable_filter; do
    uv run python scripts/train_full_map.py \
      --config "configs/candidates/${config}.yaml" \
      --features data/raw/gems-geodawn-numerical-features.tif \
      --labels data/raw/existing_faults.tif \
      --template data/raw/example_submission.tif \
      --fold-map data/processed/cv-trace-v1.tif \
      --cv-scheme trace \
      --fold $fold \
      --buffer-pixels 20 \
      --seed 20260922 \
      --output "runs/${config}-trace/fold-${fold}.tif" \
      --metrics-json "runs/${config}-trace/metrics-fold-${fold}.json"
  done
done
```

### Scoring (after training):
```bash
for config in cand_mumford_shah_log cand_structure_tensor_coherence \
              cand_ridge_valley_response cand_steerable_filter; do
  # Fault CV
  uv run python scripts/score_cv.py \
    --truth data/raw/existing_faults.tif \
    --fold-map data/processed/cv-fault-v1.tif \
    --scheme fault \
    --prediction-pattern "runs/${config}-fault/fold-{fold}.tif" \
    --output-json "runs/${config}-fault/fault.json"
  
  # Trace CV
  uv run python scripts/score_cv.py \
    --truth data/raw/existing_faults.tif \
    --fold-map data/processed/cv-trace-v1.tif \
    --scheme trace \
    --prediction-pattern "runs/${config}-trace/fold-{fold}.tif" \
    --output-json "runs/${config}-trace/trace.json"
done
```

### Three-view gate evaluation:
```bash
for config in cand_mumford_shah_log cand_structure_tensor_coherence \
              cand_ridge_valley_response cand_steerable_filter; do
  uv run python scripts/verify_candidate.py \
    --candidate "${config}" \
    --incumbent-spatial results/runs/ref-cv-spatial-full/spatial.json \
    --incumbent-fault <baseline-fault.json> \
    --incumbent-trace <baseline-trace.json> \
    --candidate-spatial "runs/${config}/spatial.json" \
    --candidate-fault "runs/${config}-fault/fault.json" \
    --candidate-trace "runs/${config}-trace/trace.json" \
    --config "configs/candidates/${config}.yaml"
done
```

## Impact

✅ Lineament candidates unblocked for fault/trace CV  
✅ Three-view gate evaluation now possible  
✅ No changes to spatial CV behavior  
✅ All tests pass, protocol proven leakage-safe

## Pull Request

**Branch**: `cursor/lineament-fault-trace-cv-protocol-3a7e`  
**PR**: https://github.com/Svyable/GEMS-CONTEST/pull/36 (draft)  
**Status**: Ready for review

## Next Steps

1. **Merge PR** after review
2. **Train candidates**: Run all four lineament candidates on fault and trace CV (commands above)
3. **Score**: Use `scripts/score_cv.py` to compute fold-wise scores
4. **Gate**: Use `scripts/verify_candidate.py` with all three views
5. **Promote**: Accept candidates that beat incumbent on ≥2 of 3 views

## Evidence

From real-data CPU runs (not reproducible in CI):
- Reference U-Net spatial CV: 0.0963 raw, 0.1230 calibrated
- Fault CV folds so far: 0.0374, 0.0449
- **Fault generalization is the weak point** — exactly where better features should help

This protocol enables testing that hypothesis honestly across all three CV views.
