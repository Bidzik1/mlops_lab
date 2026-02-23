import os
import pandas as pd


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
    