from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import subprocess
import logging
import zipfile
import os
import glob

def download_dataset():
    os.environ['KAGGLE_CONFIG_DIR'] = '/home/chris/.kaggle/'
    dataset_path = '/home/chris/MLBootcamp/dataset/'
    zip_path = dataset_path + '/global-weather-repository.zip'
    
    try:
        files = glob.glob(dataset_path + '/*')
        for f in files:
            os.remove(f)
        logging.info("Old files deleted")
        
        subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', 'nelgiriyewithana/global-weather-repository', '-p', dataset_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logging.info("Dataset downloaded")

        # Descomprimir el archivo
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        logging.info("Dataset unzipped")

    except subprocess.CalledProcessError as e:
        logging.error("Error downloading dataset")
        logging.error(e.output)
    except zipfile.BadZipFile:
        logging.error("Error unzipping dataset")

# DAG config
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 28),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG('weather_data_download',
        default_args=default_args,
        description='Download weather data daily from Kaggle',
        schedule_interval=timedelta(days=1),
        tags=['Dataset download']
        )

download_task = PythonOperator(
    task_id='download_weather_data',
    python_callable=download_dataset,
    dag=dag,
)

download_task
