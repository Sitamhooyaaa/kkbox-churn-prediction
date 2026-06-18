import json
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODELING_DATA_PATH = (
    PROJECT_DIR / "data" / "processed" / "kkbox_modeling_data_v1.csv"
)
OUTPUT_PATH = PROJECT_DIR / "examples" / "sample_request.json"


data_sample = pd.read_csv(
    MODELING_DATA_PATH,
    nrows=5000,
)

feature_sample = (
    data_sample
    .drop(columns=["msno", "is_churn"])
    .dropna()
    .head(1)
)

if feature_sample.empty:
    raise ValueError("No complete customer row found in the sample.")

sample_request = json.loads(
    feature_sample.to_json(orient="records")
)[0]

OUTPUT_PATH.write_text(
    json.dumps(sample_request, indent=2),
    encoding="utf-8",
)

print("Saved:", OUTPUT_PATH)
print(json.dumps(sample_request, indent=2))