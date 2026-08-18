# EMG Gesture Classifier

A machine learning pipeline for classifying hand/wrist gestures from surface EMG
signals, built on the [UCI EMG dataset](https://archive.ics.uci.edu/dataset/481) (36
subjects, 8-channel MYO armband, 8 gesture classes). Time-domain + autoregressive
features are extracted from sliding windows of the raw signal, class imbalance is
handled with a resampling strategy (undersampling + BorderlineSMOTE + ENN), and
gesture classification is done with either a scikit-learn MLP or a TensorFlow
neural network.

This repo covers gesture classification only. (A related EMG biometric
user-identification project lives in a separate repository.)

## Getting Started

### Installation

```bash
python -m venv venv
source venv/bin/activate    # Windows: .\venv\Scripts\activate
pip install -e .            # editable install of the emg_classifier package
```

`requirements.txt` is also provided for a pinned, non-editable environment
(`pip install -r requirements.txt`).

### Data

Place the raw UCI EMG dataset under `data/raw/UCI/<subject_id>/*.txt` (two session
files per subject). `data/`, `experiments/`, `hyperparameter_tuning/`, and
`results/` are all gitignored — they're generated/local, not part of the repo.

## Project Structure

- `src/emg_classifier/` - the installable package: data loading, feature
  extraction, models, and pipelines
  - `dataloader/` - loads and preprocesses the raw UCI EMG files
  - `features/` - sliding-window feature extraction (time-domain + AR coefficients)
  - `models/` - scikit-learn / TensorFlow / ensemble classifiers
  - `pipelines/` - orchestrates load -> extract -> split -> balance -> train
- `scripts/` - CLI entry points (`run_experiment.py`, `analyze_results.py`)
- `config/` - experiment configuration (`uci_config.yaml`)
- `tests/` - pytest suite
- `docs/report/` - the dissertation this pipeline implements Chapter 4 of, plus the
  1D-feature-set reference paper (Raurale et al., 2018) it's built on
- `legacy/` - superseded prototype scripts, kept for reference (see its own README)

## Running Experiments

Run scripts from the repo root so relative paths in `config/uci_config.yaml`
(`data/`, `results/`, `experiments/`) resolve correctly.

### Test Mode (Single Configuration)

```bash
python scripts/run_experiment.py --test
```

### Full Mode (All Configurations)

Runs every combination of channel subset, window size, and model type defined in
the config:

```bash
python scripts/run_experiment.py
```

### Hyperparameter Tuning

```bash
python scripts/run_experiment.py --tune
```

### Analyzing Results

```bash
python scripts/analyze_results.py               # most recent results/experiment_results_full_*.csv
python scripts/analyze_results.py --report       # also emit a markdown report
python scripts/analyze_results.py --results-file results/experiment_results_full_20250101_000000.csv
```

## Configuration

`config/uci_config.yaml` controls experiment parameters:

- `data.channel_configs` - EMG channel subsets to test (4-8 channels)
- `data.segment_dims` / `data.increment_dims` - sliding window sizes and their
  corresponding step sizes
- `model.types` - model types to evaluate (`sklearn`, `tensorflow`)
- `processing.use_lda` - optional LDA dimensionality reduction after scaling
- `processing.split_strategy` - evaluation protocol, see below

## Evaluation Methodology

Two things are true regardless of which `split_strategy` is used:

1. **Sliding windows are built independently per recording session/file**, and never
   slide across the boundary between two files - so a window can't blend two
   unrelated recordings into one feature vector.
2. **Class balancing (undersampling + BorderlineSMOTE + ENN) is applied to the
   training fold only, after the split, never before** - so no synthetic sample is
   ever derived from a neighbor that ends up in the test set.

`processing.split_strategy` in `config/uci_config.yaml` then selects how train/test
are divided (see `BasePipeline._prepare_data` for the implementation):

- **`subject_independent`** (default): held-out subjects are entirely unseen.
  Tests generalization to a brand-new person with zero calibration.
- **`block_holdout`**: same subjects appear in both train and test, held out by
  whole recording session. In this dataset each subject's two sessions are their
  left/right arm, so this tests cross-arm transfer within the same person - it does
  *not* hold sensor placement fixed.
- **`intra_session`**: the same recording session is split by time, computed
  *per class* (gesture classes are presented in a rolling, progressively-later
  sequence within a session, so a single global time cut starves whichever class
  finishes earliest of any test examples - see the method's docstring). This holds
  the physical sensor placement fixed and is the closest match to the reference
  paper's own train/test-session protocol.

All three avoid the two leakage bugs above and are window-overlap-safe (no train
window ever shares a raw sample with a test window).

**Finding:** all three protocols land in the same **63-65% accuracy** band on the
8-channel/2048-sample-window config (vs. this pipeline's own ~99% before the fix,
and the reference paper's 95.57% on its own 10-subject, per-subject-calibrated,
same-placement dataset). Since switching *who* is held out barely moves the number,
the gap is not primarily an artifact of the split's subject/session philosophy - the
leakage bugs were the dominant cause of the original inflated figures, and a real,
unresolved generalization gap remains beyond that (most plausibly sensitivity to the
Myo armband's uncalibrated, per-donning placement - see the `intra_session` result,
where even holding placement fixed didn't close the gap - though that run's test
folds were small and single-split, not multi-seed averaged, so this isn't
conclusive). Improving on 63-65% would need work on the feature representation or
model architecture, not further evaluation-protocol changes.

Older result tables/plots in `docs/results/` (if present) predate this fix and
should not be treated as current - they were produced by a resample-before-split
evaluation that leaked information between train and test. Regenerate results with
`scripts/run_experiment.py` before citing any accuracy numbers from this repo.

## Results

Experiment results are saved to CSV files in `results/`, including accuracy, F1,
precision, recall, and timing, alongside a per-run log file.
