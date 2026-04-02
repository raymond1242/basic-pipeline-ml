# Sleep Health ML Pipeline

Binary classification pipeline that predicts whether a person **felt rested** (`felt_rested`) based on sleep, lifestyle, and health metrics.

## Project Structure

```
basic-pipeline-ml/
├── main.py            # Pipeline orchestrator
├── config.py          # Feature columns and target definition
├── preprocessing.py   # Data loading, cleaning, encoding, feature selection
├── training.py        # Train/test split, scaling, model training, evaluation
├── monitoring.py      # Data quality checks (raw and processed)
├── requirements.in    # Top-level dependencies
└── requirements.txt   # Pinned dependencies (generated)
```

## Installation

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install pip-tools and compile pinned dependencies
pip install pip-tools
pip-compile requirements.in

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

The dataset is loaded directly from Google Drive. To use a local copy instead, update `DATA_PATH` in `main.py`.

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 1 | `preprocessing.py` | Load CSV, drop missing values, label-encode 6 categorical columns |
| 2 | `monitoring.py` | Log raw data quality: shape, nulls per column, duplicates, statistics |
| 3 | `preprocessing.py` | Select the 27 feature columns + target |
| 4 | `monitoring.py` | Log processed data quality: shape, types, target distribution |
| 5 | `training.py` | Stratified train/test split, scale features, train RandomForest, evaluate |

## Model

- **Algorithm**: RandomForestClassifier
- **Hyperparameters**: `n_estimators=200`, `max_depth=12`, `min_samples_leaf=5`
- **Target**: `felt_rested` (binary)
- **Split**: 80/20 with stratification

## Design Decision: Scaling in Training, Not Preprocessing

`StandardScaler` is applied **after** the train/test split inside `training.py`, not in `preprocessing.py`. This is intentional.

The scaler computes the mean and standard deviation of each feature. If scaling is done before splitting, those statistics are calculated using the entire dataset — including the test set. The model then indirectly "sees" test data during training, which is known as **data leakage**. This inflates evaluation metrics and gives an unrealistic picture of how the model will perform on truly unseen data.

The correct approach:

1. **Split** into train and test sets
2. **Fit** the scaler on the training set only (`fit_transform`)
3. **Transform** the test set using the training set's statistics (`transform`)

This ensures the test set is evaluated under conditions that mirror production, where new data arrives without prior knowledge of its distribution.

## Dependencies

- pandas
- numpy
- scikit-learn
