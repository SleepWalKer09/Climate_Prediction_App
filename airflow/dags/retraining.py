from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import json
import subprocess
import os

BASE_DIR = os.getenv('MLBOOTCAMP_HOME', '/home/chris/MLBootcamp')

def preprocess():
    script_path = os.path.join(BASE_DIR, 'scripts', 'Preprocess.py')
    result = subprocess.run(['python3', script_path], capture_output=True, text=True)
    #resultado de ejecucion
    if result.returncode == 0:
        print("Preprocesamiento de datos terminado correctamente.")
        if result.stdout:
            print("Salida del script:", result.stdout)
    else:
        print("Error en la ejecución del script de preprocesamiento.")
        if result.stderr:
            print("Error:", result.stderr)

def train_model():
    script_path = '/home/chris/MLBootcamp/scripts/Model_train.py'
    result = subprocess.run(['python3', script_path], capture_output=True, text=True)
    #resultado de ejecucion
    if result.returncode == 0:
        print("El modelo ha sido entrenado correctamente")
        print("Metadatos guardados correctamente")
        if result.stdout:
            print("Salida del script:", result.stdout)
    else:
        print("Error en la ejecución del script de Entrenamiento.")
        if result.stderr:
            print("Error:", result.stderr)

def compare_and_update_model():
    metrics_file_path = '/home/chris/MLBootcamp/model_data/metrics.json'
    with open(metrics_file_path, 'r') as file:
        new_metrics = json.load(file)
    #carga metricas del modelo desplegado
    current_metrics_file_path = '/home/chris/MLBootcamp/model_data/metrics_current.json'
    try:
        with open(current_metrics_file_path, 'r') as file:
            current_metrics = json.load(file)
    except FileNotFoundError:
        current_metrics = None

    # Comparar métricas y decide si actualizar el modelo
    metrics_better = 0
    if current_metrics:
        for metric in new_metrics:
            if metric in ['MeanSquaredError', 'MeanAbsoluteError', 'MeanSquaredLogarithmicError', 'RootMeanSquaredError']:
                # Para las 4 métricas, un valor más bajo es mejor!!
                if new_metrics[metric] < current_metrics[metric]:
                    metrics_better += 1

    # Si 3 de 4 métricas son mejores o no hay métricas actuales, actualiza el modelo
    if metrics_better >= 3 or current_metrics is None:
        print("El modelo recien entrenado es mejor y será desplegado")
        # Actualiza el archivo de métricas actuales
        with open('/home/chris/MLBootcamp/model_data/metrics_current.json', 'w') as file:
            json.dump(new_metrics, file, indent=4)
    else:
        print("El modelo desplegado continúa siendo el mejor. Eliminando el modelo recién entrenado.")
        # Encuentra el archivo del modelo recién entrenado basado en la fecha más reciente
        latest_model_file = max([f for f in os.listdir("/home/chris/MLBootcamp/model_data/") if f.startswith('model_GRU_')], key=lambda x: os.path.getmtime(os.path.join("/home/chris/MLBootcamp/model_data/", x)))
        os.remove(os.path.join("/home/chris/MLBootcamp/model_data/", latest_model_file))
        print(f"Modelo {latest_model_file} eliminado.")


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG('weekly_model_retraining',
        default_args=default_args,
        description='Weekly retraining of the weather prediction model',
        tags=['Weekly model training'],
        schedule_interval='@weekly')

preprocess = PythonOperator(
    task_id='preprocess',
    python_callable=preprocess,
    dag=dag,
)

train_model = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

compare_and_update_model = PythonOperator(
    task_id='compare_and_update_model',
    python_callable=compare_and_update_model,
    dag=dag,
)

preprocess >> train_model >> compare_and_update_model