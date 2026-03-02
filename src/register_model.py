import mlflow
from mlflow.tracking import MlflowClient

def main():
    mlflow.set_tracking_uri("file:///opt/mlops_lab_1/mlruns")

    client = MlflowClient()
    experiment = client.get_experiment_by_name("House_Rent_Prediction")

    if experiment is None:
        raise Exception("Experiment not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise Exception("No MLflow runs found.")

    latest_run = runs[0]
    run_id = latest_run.info.run_id

    model_uri = f"runs:/{run_id}/random_forest_model"

    mlflow.register_model(model_uri, "HouseRentModel")

    print("Model registered successfully in MLflow Model Registry.")


if __name__ == "__main__":
    main()