import argparse
import mlflow
import mlflow.sklearn
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from data_preprocessing import load_data, preprocess_data
from utils import regression_metrics, plot_feature_importance


def parse_args():
    parser = argparse.ArgumentParser(description="Train RandomForest for House Rent Prediction")

    parser.add_argument("--data_path", type=str,
                        default="data/raw/House_Rent_10M_balanced_40cities.csv")

    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=365)

    parser.add_argument("--experiment_name", type=str,
                        default="House_Rent_Prediction")

    parser.add_argument("--author", type=str, default="Nazar")
    parser.add_argument("--dataset_version", type=str, default="v1")

    return parser.parse_args()


def main():
    args = parse_args()
    df = load_data(args.data_path)
    df = df.sample(n=1000000, random_state=365)
    X, y, preprocessor = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state
    )

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
        mlflow.log_param("test_size", args.test_size)
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

        #Логування метрик (TRAIN)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("train_r2", train_r2)

        #Логування метрик (TEST)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_r2", test_r2)

        #Логування моделі
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
