"""ML Pipeline: Sleep Health — predict felt_rested from sleep metrics."""

import logging

import preprocessing as pp
import monitoring as mn
import training as tr

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

DATA_PATH = "sleep_health_dataset.csv"
TARGET_COL = "felt_rested"

FEATURE_COLS = [
    "sleep_duration_hrs",
    "sleep_quality_score",
    "rem_percentage",
    "deep_sleep_percentage",
    "sleep_latency_mins",
    "wake_episodes_per_night",
    "stress_score",
    "work_hours_that_day",
    "caffeine_mg_before_bed",
    "alcohol_units_before_bed",
    "screen_time_before_bed_mins",
    "exercise_day",
    "steps_that_day",
    "nap_duration_mins",
    "bmi",
    "age",
    "heart_rate_resting_bpm",
    "sleep_aid_used",
    "shift_work",
    "room_temperature_celsius",
    "weekend_sleep_diff_hrs",
    "gender_enc",
    "occupation_enc",
    "chronotype_enc",
    "mental_health_condition_enc",
    "season_enc",
    "day_type_enc",
]


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
