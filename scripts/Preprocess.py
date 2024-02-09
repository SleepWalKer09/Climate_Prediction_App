import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np
import joblib
import os

BASE_DIR = os.getenv('MLBOOTCAMP_HOME', '/home/chris/MLBootcamp')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
DATA_DIR = os.path.join(BASE_DIR, 'model_data')
dataset_path = os.path.join(DATASET_DIR, 'GlobalWeatherRepository.csv')
input_features_path = os.path.join(DATA_DIR, 'input_features.npy')
target_features_path = os.path.join(DATA_DIR, 'target_features.npy')
encoder_path = os.path.join(DATA_DIR, 'encoder.joblib')
scaler_path = os.path.join(DATA_DIR, 'scaler.joblib')

df = pd.read_csv(dataset_path)
print("dataset loaded para preprocesamiento!!")

# Convertir columnas de fecha/hora y Extracción de características temporales
df['last_updated'] = pd.to_datetime(df['last_updated'])
df['hour'] = df['last_updated'].dt.hour
df['day'] = df['last_updated'].dt.day
df['month'] = df['last_updated'].dt.month

#objetivos
target_features = df[[
    'temperature_celsius', 'wind_mph', 'pressure_mb', 'humidity', 'uv_index',
    'air_quality_us-epa-index', 'cloud',
    'air_quality_Carbon_Monoxide', 'air_quality_Ozone', 'gust_mph', 'gust_kph'
]]

encoder = OneHotEncoder(sparse_output=False)# Codificación One-Hot para datos categóricos
encoded_categorical = encoder.fit_transform(df[['country', 'location_name']])
scaler = MinMaxScaler()# Normalizar características numéricas
scaled_numerical = scaler.fit_transform(df[['latitude', 'longitude', 'hour', 'day', 'month']])

# Combinar datos codificados y normalizados, excluyendo las características objetivo
input_features = np.concatenate([encoded_categorical, scaled_numerical], axis=1)

np.save(input_features_path, input_features)
np.save(target_features_path, target_features.values)
joblib.dump(encoder, encoder_path)
joblib.dump(scaler, scaler_path)

print("Datos preprocesados guardados!!")