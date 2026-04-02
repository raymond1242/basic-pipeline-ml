"""Pipeline configuration: feature columns and target."""

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
