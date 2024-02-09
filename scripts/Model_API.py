from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import glob
import os

app = FastAPI()

# Función para buscar el modelo más reciente
def cargar_modelo_mas_reciente():
    lista_modelos = glob.glob('../model_data/model_GRU_*.keras')
    modelo_mas_reciente = max(lista_modelos, key=os.path.getctime)
    return load_model(modelo_mas_reciente)

dataset = pd.read_csv('../dataset/GlobalWeatherRepository.csv')
encoder = joblib.load('../model_data/encoder.joblib')
scaler = joblib.load('../model_data/scaler.joblib')
model = cargar_modelo_mas_reciente()

class PrediccionRequest(BaseModel):
    ubicacion_usuario: str
    pais_usuario: str
    fecha_hora_usuario: str

# Función para buscar coordenadas en el dataset
def buscar_coordenadas(dataset, ciudad, pais):
    registro = dataset[(dataset['location_name'] == ciudad) & (dataset['country'] == pais)]
    if not registro.empty:
        return registro.iloc[0]['latitude'], registro.iloc[0]['longitude']
    else:
        return None, None

def realizar_prediccion(ubicacion_usuario, pais_usuario, fecha_hora_usuario):
    input_data = {
        'country': [pais_usuario],
        'location_name': [ubicacion_usuario],
        'last_updated': pd.to_datetime([fecha_hora_usuario])
    }
    input_df = pd.DataFrame(input_data)

    # Extracción de características temporales
    input_df['hour'] = input_df['last_updated'].dt.hour
    input_df['day'] = input_df['last_updated'].dt.day
    input_df['month'] = input_df['last_updated'].dt.month

    
    latitud, longitud = buscar_coordenadas(dataset, ubicacion_usuario, pais_usuario)
    if latitud is None or longitud is None:
        return "Ubicación no encontrada en el dataset"

    # Codificación One-Hot y normalización
    encoded_categorical = encoder.transform(input_df[['country', 'location_name']])
    scaled_numerical = scaler.transform([[latitud, longitud, input_df['hour'][0], input_df['day'][0], input_df['month'][0]]])

    # Combinar datos codificados y normalizados para crear la secuencia de entrada
    processed_input = np.concatenate([encoded_categorical, scaled_numerical], axis=1)
    input_sequence = np.array([processed_input] * 5)  # Crear una secuencia
    input_sequence = input_sequence.reshape((1, input_sequence.shape[0], input_sequence.shape[2]))

    prediction = model.predict(input_sequence)

    return prediction



@app.get("/predict/")
def predict(ubicacion_usuario: str, pais_usuario: str, fecha_hora_usuario: str):
    prediccion = realizar_prediccion(ubicacion_usuario, pais_usuario, fecha_hora_usuario)
    print(f"Modelo:",{model})
    if isinstance(prediccion, str):
        return {"error": prediccion}

    nombres_variables = [
        'Temperatura Celsius', 'Velocidad del viento mph', 'Presión atmosférica mb',
        'Humedad relativa %', 'Índice UV', 'Calidad del aire EPA EE.UU.',
        'Cobertura de nubes %', 'Monóxido de Carbono', 'Ozono', 
        'Ráfaga de viento mph', 'Ráfaga de viento kph'
    ]

    resultados = {
        #"Predicción para": f"{ubicacion_usuario}, {pais_usuario} en {fecha_hora_usuario}"
    }
    for i, nombre_var in enumerate(nombres_variables):
        resultados[nombre_var] = f"{prediccion[0][i]:.2f}"

    return resultados
