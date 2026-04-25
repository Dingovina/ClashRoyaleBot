# ML Roadmap (ClashRoyaleBot)

## Current Model Assessment

Based on the current training/evaluation scripts and local checkpoints:

- `card_cnn.pt` (`CardHandNet`, 9 classes) on `data/processed/cards`: **119/119 = 100%**
- `elixir_cnn.pt` (`ElixirDigitNet`, 11 classes) on `data/processed/elixir_test`: **23/23 = 100%**
- `battlefield_cnn.pt` (`BattlefieldScreenNet`, binary) on `data/processed/battlefield_test`: **17/17 = 100%**
- Battlefield threshold sweep (`0.5`, `0.65`, `0.8`, `0.95`) also stays at **100%** on the same dataset

Checkpoint metadata indicates very small datasets:

- Battlefield: train 13 / val 4
- Elixir: train 12 / val 11
- Cards: train 89 / val 30

## Key Reliability Risks

- **Likely optimistic metrics:** evaluations are run on tiny in-domain sets, close to training distribution.
- **High overfitting risk:** sample counts are too small for robust generalization claims.
- **Data contract mismatch:** some scripts expect fullscreen images while processed datasets already contain ROI crops.
  - Example: `eval_battlefield_classifier.py` fails on ROI-only `data/processed/battlefield_test`.
- **Incomplete evaluation tooling:** no dedicated `scripts/eval/eval_elixir_classifier.py`.
- **Class imbalance in cards:** e.g. `empty` has very few examples compared to frequent classes.
- **Confidence calibration gap:** runtime behavior depends on thresholds, but confidence calibration is not formalized.

## Improvement Plan (Prioritized)

### P0 (Must Do First)

1. **Establish strict train/val/test split by match/session**, not by frame.
2. **Unify data contract** across crop/train/eval/runtime:
   - either all training/eval scripts consume fullscreen and crop internally,
   - or all consume pre-cropped ROI and never crop again.
3. **Add missing eval pipeline for elixir** and standardize reports:
   - accuracy
   - per-class recall
   - confusion matrix
4. **Create a holdout set** (`data/holdout/*`) excluded from training and hyperparameter tuning.

### P1 (High Impact Next)

1. **Scale dataset volume 5-10x** with broader visual variation:
   - brightness and contrast variation
   - motion blur/compression artifacts
   - transitions between menus, replay, game over, etc.
2. **Address class imbalance**:
   - weighted loss and/or balanced sampling
   - explicit target counts for rare classes (cards and elixir states)
3. **Calibrate confidence outputs** (e.g. temperature scaling) and tune runtime thresholds on holdout metrics.

### P2 (Model/Training Quality)

1. Add learning-rate scheduling and early stopping on validation metric.
2. Add label smoothing for multi-class tasks (cards/elixir).
3. Improve augmentations for realism (JPEG artifacts, gamma shifts, mild blur/noise).
4. Track repeated experiments with manifest snapshots for reproducibility.

## Execution Plan (1-2 Days)

1. Implement unified eval scripts: `eval_card`, `eval_elixir`, `eval_battlefield`.
2. Introduce fixed split definition (`data/splits.yaml`) by match/session ids.
3. Generate standardized JSON reports and artifact manifests for every run.
4. Recompute baseline metrics on true holdout.
5. Retune runtime thresholds (`battlefield_score_threshold`, `battlefield_end_score_threshold`, action confidence thresholds) using holdout evidence.

## Success Criteria

- Metrics remain stable on holdout after data/contract cleanup.
- Evaluation is reproducible and script-driven for all three models.
- Runtime thresholds are justified by calibrated model confidence, not ad-hoc tuning.
- Model quality improves without increasing false actions in live runtime.
