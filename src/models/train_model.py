from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"

MODELING_DATA_PATH = PROCESSED_DIR / "kkbox_modeling_data_v1.csv"
MODEL_OUTPUT_PATH = MODELS_DIR / "hgb_churn_model_v1.joblib"
TEST_RESULTS_PATH = PROCESSED_DIR / "selected_model_test_results_v1_from_script.csv"

TARGET_COL = "is_churn"
ID_COL = "msno"
SELECTED_THRESHOLD = 0.13


def build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_preprocessor, numeric_features),
            ("categorical", categorical_preprocessor, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=150,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    l2_regularization=0.1,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_predictions(y_true, y_proba, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fp_cost = 1
    fn_cost = 5
    cost_per_customer = ((fp * fp_cost) + (fn * fn_cost)) / len(y_true)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "brier_score": brier_score_loss(y_true, y_proba),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "cost_per_customer": cost_per_customer,
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading modeling data...")
    modeling_data = pd.read_csv(MODELING_DATA_PATH)

    X = modeling_data.drop(columns=[TARGET_COL, ID_COL])
    y = modeling_data[TARGET_COL]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=42,
        stratify=y,
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    categorical_features = [
        "gender",
        "city",
        "registered_via",
        "latest_payment_method_id",
        "latest_is_auto_renew",
        "latest_is_cancel",
    ]
    numeric_features = [col for col in X_train.columns if col not in categorical_features]

    print("Training selected HGB model...")
    model = build_pipeline(numeric_features, categorical_features)
    model.fit(X_train, y_train)

    print("Evaluating validation and test sets...")
    valid_proba = model.predict_proba(X_valid)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    valid_results = evaluate_predictions(y_valid, valid_proba, SELECTED_THRESHOLD)
    test_results = evaluate_predictions(y_test, test_proba, SELECTED_THRESHOLD)

    results = pd.DataFrame(
        [
            {"split": "validation", **valid_results},
            {"split": "test", **test_results},
        ]
    )
    results.to_csv(TEST_RESULTS_PATH, index=False)

    artifact = {
        "model": model,
        "threshold": SELECTED_THRESHOLD,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target_col": TARGET_COL,
        "id_col": ID_COL,
    }
    joblib.dump(artifact, MODEL_OUTPUT_PATH)

    print(f"Saved model: {MODEL_OUTPUT_PATH}")
    print(f"Saved results: {TEST_RESULTS_PATH}")
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
