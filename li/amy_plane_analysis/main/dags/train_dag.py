from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

import os

PROJECT_DIR = "/Users/moomintroll/Desktop/tsk/Diagnostics_of_cardiovascular_diseases"

with DAG(
    dag_id="random_forest_training",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["ml", "random_forest"],
) as dag:

    generate_features = BashOperator(
        task_id="generate_features",
        bash_command=f"python {PROJECT_DIR}/process_ecg.py",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"python {PROJECT_DIR}/classification/train_random_forest.py",
    )

    generate_features >> train_model