# EMG Classifier

This project provides a framework for EMG signal classification using various machine learning models.

## Getting Started

### Installation

1. Create a new Python virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `.\venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

- `config/` - Configuration files for experiments
- `data/` - Data directories for raw, processed, and interim data
- `src/` - Source code
- `results/` - Experiment results will be saved here

## Running Experiments

The main entry point is `run_experiment.py`, which can be run in two modes:

### Test Mode (Single Configuration)

Run a single experiment with predefined test settings:

```
python run_experiment.py --test
```

This uses the test configuration specified in the config YAML file.

### Full Mode (All Configurations)

Run experiments for all combinations of channels, window sizes, and models:

```
python run_experiment.py
```

### Hyperparameter Tuning

To enable hyperparameter tuning before training (for TensorFlow models):

```
python run_experiment.py --tune
```

## Configuration

The `config/uci_config.yaml` file controls experiment parameters:

- **Data Configuration**: Channel subsets, segment dimensions, and data splits
- **Model Configuration**: Model types and hyperparameters
- **Path Configuration**: Directories for data and results storage

### Key Parameters

- `channel_configs`: Different EMG channel subsets to test
- `segment_dims`: Window sizes for segmentation
- `increment_dims`: Increment sizes corresponding to each window
- `model.types`: Model types to evaluate ("sklearn", "tensorflow")

## Results

Experiment results are saved to CSV files in the `results/` directory, including:
- Accuracy, F1 score, precision, and recall
- Training and inference times
- Experiment configuration details

A log file is also generated with details about each experiment run.
