from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess
import os


# =========================
# DEFAULT ARGS
# =========================

default_args = {
    "owner": "ecg_pipeline",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# =========================
# PATHS
# =========================

BASE_DIR = "/Users/moomintroll/Desktop/tsk/Diagnostics_of_cardiovascular_diseases"

INFERENCE_SCRIPT = os.path.join(BASE_DIR, "inference.py")
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")


# =========================
# TASK FUNCTION
# =========================

def run_inference():
    cmd = [
        "python",
        INFERENCE_SCRIPT,
        CONFIG_PATH
    ]

    print("Running inference pipeline...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Inference failed")


# =========================
# DAG DEFINITION
# =========================

with DAG(
    dag_id="ecg_inference_pipeline",
    default_args=default_args,
    description="ECG inference pipeline for VECG model",
    schedule_interval=None,   
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ecg", "inference", "ml"]
) as dag:

    inference_task = PythonOperator(
        task_id="run_ecg_inference",
        python_callable=run_inference
    )

    inference_task