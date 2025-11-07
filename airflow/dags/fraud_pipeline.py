
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="fraud_detection_pipeline",
    default_args=default_args,
    description="Daily scoring and weekly retraining for fraud detection",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    gen_data = BashOperator(
        task_id="generate_data",
        bash_command="cd /opt/airflow/dags/repo && python src/utils/generate_data.py --rows 50000 --out data/transactions.csv",
    )

    daily_score = BashOperator(
        task_id="daily_score",
        bash_command="echo 'Call scoring API here in a real deployment'",
    )

    weekly_retrain = BashOperator(
        task_id="weekly_retrain",
        bash_command="cd /opt/airflow/dags/repo && python src/pipelines/train.py --data data/transactions.csv --outdir artifacts",
    )

    gen_data >> daily_score >> weekly_retrain
