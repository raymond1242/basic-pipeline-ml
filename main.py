"""ML Pipeline: Sleep Health — predict felt_rested from sleep metrics."""

import logging

from config import FEATURE_COLS, TARGET_COL
import preprocessing as pp
import monitoring as mn
import training as tr

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

DATA_PATH = "https://drive.google.com/uc?id=1MlddENKEcBMTFAQbYKCaETyKkc0bgEXn"
# Local path for testing:
# DATA_PATH = "sleep_health_dataset.csv"

def main() -> None:
    """Run the full ML pipeline."""
    # 1. Load raw data
    df_raw = pp.load_data(DATA_PATH)

    # 2. Monitor raw data quality
    mn.monitor_raw(df_raw)

    # 3. Preprocess: clean, encode categoricals, select features
    df_processed = pp.preprocess(df_raw, FEATURE_COLS, TARGET_COL)

    # 4. Monitor processed data quality
    mn.monitor_processed(df_processed, target_col=TARGET_COL)

    # 5. Train and evaluate
    model, metrics = tr.train(df_processed, TARGET_COL)
    tr.evaluate(model, metrics)


if __name__ == "__main__":
    main()
