import os
import json
import pandas as pd


# PREPARE STAGE TEST
def test_prepare_stage_output():
    data_path = "data/prepared/train.csv"
    assert os.path.exists(data_path), "Prepare stage failed: train.csv not found"

    df = pd.read_csv(data_path)

    required_cols = {
        "BHK",
        "Size",
        "Bathroom",
        "Building Type",
        "Area Type",
        "Furnishing Status",
        "Tenant Preferred",
        "Point of Contact",
        "Rent",
    }

    missing = required_cols - set(df.columns)
    assert not missing, f"Missing columns: {missing}"

    assert df["Rent"].notna().all(), "Target contains NaN"
    assert df.shape[0] > 100, "Too few rows"


# TRAIN STAGE TEST
def test_train_stage_artifacts():
    model_exists = os.path.exists("data/models/model.pkl")
    metrics_exists = os.path.exists("metrics.json")
    plot_exists = os.path.exists("feature_importance.png")

    if model_exists:
        assert metrics_exists, "metrics.json not found after train"
        assert plot_exists, "feature_importance.png not found after train"


# OPTIMIZE STAGE TEST
def test_optimize_stage_artifacts():
    best_model_exists = os.path.exists("models/best_model.pkl")
    if best_model_exists:
        assert os.path.exists("metrics.json"), "metrics.json not found after optimize"


# QUALITY GATE
def test_quality_gate():
    if not os.path.exists("metrics.json"):
        return
    with open("metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    if "test_r2" in metrics:
        threshold = float(os.getenv("R2_THRESHOLD", "0.50"))
        r2 = float(metrics["test_r2"])
        assert r2 >= threshold, f"Quality Gate failed: R2={r2:.4f} < {threshold}"
    elif "final_metric" in metrics:
        threshold = float(os.getenv("RMSE_THRESHOLD", "15000"))
        rmse = float(metrics["final_metric"])
        assert rmse <= threshold, f"Quality Gate failed: RMSE={rmse:.2f} > {threshold}"
