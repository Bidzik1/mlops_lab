import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def preprocess_data(df: pd.DataFrame):
    y = df["Rent"]
    X = df.drop(columns=["Rent"])

    numeric_features = ["BHK", "Size", "Bathroom"]

    categorical_features = [
        "Building Type",
        "Area Type",
        "Furnishing Status",
        "Tenant Preferred",
        "Point of Contact",
    ]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return X, y, preprocessor
