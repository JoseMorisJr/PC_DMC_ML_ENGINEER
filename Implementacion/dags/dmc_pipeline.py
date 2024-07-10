import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from data_processing import descarga_informacion, processing
from autoML import running_model
from data_output import pipeline_test, submitt_test, post_test_kaggle




default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 3, 26),
    'email ':['moris.jose.jr@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_workflow_kaggle',
    default_args=default_args,
    description='Un pipeline para resolver una competencia de kaggle',
    schedule_interval='0 17 * * *',
)

descarga_informacion = PythonOperator(
    task_id='descarga_informacion',
    python_callable= descarga_informacion,
    dag=dag,
)

processing = PythonOperator(
    task_id='processing',
    python_callable = processing,
    dag=dag,
)

running_model = PythonOperator(
    task_id='running_model',
    python_callable = running_model,
    dag=dag,
)


pipeline_test = PythonOperator(
    task_id='pipeline_test',
    python_callable=pipeline_test,
    dag=dag,
)

submitt_test = PythonOperator(
    task_id='submitt_test',
    python_callable=submitt_test,
    dag=dag,
)


post_test_kaggle = PythonOperator(
    task_id='post_test_kaggle',
    python_callable=post_test_kaggle,
    dag=dag,
)

descarga_informacion >> processing >> running_model >> pipeline_test >> submitt_test >> post_test_kaggle