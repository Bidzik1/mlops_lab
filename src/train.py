import argparse
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from data_preprocessing import preprocess_data
from utils import regression_metrics, plot_feature_importance

import os
import pandas as pd
import joblib
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Train RandomForest for House Rent Prediction")

    # DVC paths
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")

    # Model hyperparameters
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=365)

    # MLflow metadata
    parser.add_argument("--experiment_name", type=str,
                        default="House_Rent_Prediction")

    parser.add_argument("--author", type=str, default="Nazar")
    parser.add_argument("--dataset_version", type=str, default="v1")

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.input_dir, "train.csv")
    test_path = os.path.join(args.input_dir, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train, y_train, preprocessor = preprocess_data(train_df)
    X_test, y_test, _ = preprocess_data(test_df)

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run():

        #Логування параметрів
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("random_state", args.random_state)

        #Логування тегів
        mlflow.set_tag("author", args.author)
        mlflow.set_tag("dataset_version", args.dataset_version)
        mlflow.set_tag("model_type", "RandomForest")

        #Навчання
        pipeline.fit(X_train, y_train)

        #Train predictions
        y_train_pred = pipeline.predict(X_train)
        train_rmse, train_r2 = regression_metrics(y_train, y_train_pred)

        #Test predictions
        y_test_pred = pipeline.predict(X_test)
        test_rmse, test_r2 = regression_metrics(y_test, y_test_pred)

        metrics = {
            "train_rmse": float(train_rmse),
            "train_r2": float(train_r2),
            "test_rmse": float(test_rmse),
            "test_r2": float(test_r2)
        }

        with open("metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        #Логування метрик (TRAIN)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("train_r2", train_r2)

        #Логування метрик (TEST)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_r2", test_r2)

        #Логування моделі
        joblib.dump(pipeline, os.path.join(args.output_dir, "model.pkl"))
        mlflow.sklearn.log_model(pipeline, "random_forest_model")

        feature_names = (
            pipeline.named_steps["preprocessor"]
            .get_feature_names_out()
        )

        plot_feature_importance(
            pipeline.named_steps["model"],
            feature_names,
            output_path="feature_importance.png"
        )

        mlflow.log_artifact("feature_importance.png")

        print("Training completed.")
        print(f"Train RMSE: {train_rmse:.2f}")
        print(f"Test RMSE: {test_rmse:.2f}")
        print(f"Train R2: {train_r2:.4f}")
        print(f"Test R2: {test_r2:.4f}")


if __name__ == "__main__":
    main()
