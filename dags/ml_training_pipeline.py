from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

import json


def evaluate_model(**kwargs):
    """
    Читає metrics.json та передає метрики в XCom
    """
    with open("/opt/mlops_lab_1/metrics.json", "r") as f:
        metrics = json.load(f)

    return metrics


def check_accuracy(**kwargs):
    """
    Branching logic: якщо test_r2 > 0.6 → реєструємо модель
    """
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="evaluate_model")

    if metrics["test_r2"] > 0.6:
        return "register_model"
    return "stop_pipeline"


with DAG(
    dag_id="ml_training_pipeline",
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
    tags=["mlops"],
) as dag:

    check_data = BashOperator(
        task_id="check_data",
        bash_command="test -f /opt/mlops_lab_1/data/prepared/train.csv",
    )

    prepare = BashOperator(
        task_id="prepare_data",
        bash_command="cd /opt/mlops_lab_1 && dvc repro prepare",
    )

    train = BashOperator(
        task_id="train_model",
        bash_command="cd /opt/mlops_lab_1 && dvc repro train",
    )

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    branch = BranchPythonOperator(
        task_id="branch_decision",
        python_callable=check_accuracy,
    )

    register = BashOperator(
        task_id="register_model",
        bash_command="cd /opt/mlops_lab_1 && python src/register_model.py",
    )

    stop = EmptyOperator(task_id="stop_pipeline")

    check_data >> prepare >> train >> evaluate >> branch
    branch >> register
    branch >> stop