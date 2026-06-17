# HiCInterpolate Architecture Update

## Changed

- Capped pyramid cost-volume search to avoid huge full-resolution windows. Previous finest level used `max_disp=32`, creating `65x65 = 4225` cost channels at `128x128`. New schedule keeps each level small.
- Made flow prediction signed by removing final `BatchNorm2d(2)` and `ReLU`. Flow needs positive and negative displacement.
- Replaced final two-channel projection with sparse-aware output head: support branch, intensity branch, residual branch.
- Added input support prior from `max(x0, x2)` plus local support gating so known sparse contacts are preserved without allowing dense mask activation everywhere.
- Added diagonal-distance channel so output head can learn genomic-distance structure.
- Enforced symmetry on both predicted contact map and predicted support mask.
- Removed double sigmoid in training loss. Model output is already in `[0, 1]`, and train/validation now use same output contract.

## Expected Effect

- Faster training from smaller cost volumes.
- Sharper sparse maps from support-gated intensity and residual prediction.
- Less oversmoothing because final output is not only a smooth decoder projection.
- Better SSIM/SCC/HiCRep potential because diagonal structure, support, and symmetry are explicit in architecture.

## Files

- `src/flow_predictor/model.py`
- `src/interpolator/model.py`
- `src/train_lib.py`
