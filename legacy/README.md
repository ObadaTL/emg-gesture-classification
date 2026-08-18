# Legacy Code

This directory contains early prototype scripts from the initial development phase of the Final Year Project. They were written as standalone experiments before the codebase was refactored into the modular pipeline under `src/`.

## Files

| File | Description |
|------|-------------|
| `utils.py` | Shared utilities for feature extraction and model training |
| `data_handler.py` | Early data loading logic |
| `id_classifier.py` | Initial gesture classifier prototype |
| `idWHPClassifier.py` | Wrist/hand pose classifier variant |
| `wrist_hand_classifier.py` | Wrist and hand classification experiment |
| `hand_pose_classification.py` | Hand pose classification script |
| `combine_files.py` | Utility to merge raw data files |

## Status

**Superseded.** Use `scripts/run_experiment.py` and the modules in `src/emg_classifier/` for all new work. These files are kept for reference and to document the project's evolution.
