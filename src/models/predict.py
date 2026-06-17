import argparse
from pathlib import Path

import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_DIR / "models" / "hgb_churn_model_v1.joblib"
DEFAULT_INPUT_PATH = PROJECT_DIR / "data" / "processed" / "kkbox_modeling_data_v1.csv"
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "data" / "processed" / "churn_predictions_v1.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate KKBox churn predictions from a processed modeling table."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    artifact = joblib.load(args.model_path)
    model = artifact["model"]
    threshold = artifact["threshold"]
    target_col = artifact["target_col"]
    id_col = artifact["id_col"]

    data = pd.read_csv(args.input_path)
    feature_data = data.drop(columns=[col for col in [target_col, id_col] if col in data.columns])

    probabilities = model.predict_proba(feature_data)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    output = pd.DataFrame(
        {
            id_col: data[id_col] if id_col in data.columns else range(len(data)),
            "churn_probability": probabilities,
            "prediction_threshold": threshold,
            "predicted_churn": predictions,
        }
    )

    if target_col in data.columns:
        output[target_col] = data[target_col]

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_path, index=False)

    print(f"Saved predictions: {args.output_path}")
    print(f"Rows: {len(output)}")
    print("Predicted churn rate:")
    print(output["predicted_churn"].mean().round(4))


if __name__ == "__main__":
    main()
