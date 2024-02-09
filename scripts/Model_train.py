from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError, MeanSquaredLogarithmicError, RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
import numpy as np
import json
import os

BASE_DIR = os.getenv('MLBOOTCAMP_HOME', '/home/chris/MLBootcamp')
DATA_DIR = os.path.join(BASE_DIR, 'model_data')
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')

input_features_path = os.path.join(DATA_DIR, 'input_features.npy')
target_features_path = os.path.join(DATA_DIR, 'target_features.npy')
model_save_path = os.path.join(DATA_DIR, f"model_GRU_{datetime.now().strftime('%Y%m%d')}.keras")
metrics_save_path = os.path.join(DATA_DIR, 'metrics.json')

def create_sequences(input_data, target_data, window_size):
    sequences = []
    labels = []
    for i in range(len(input_data) - window_size):
        sequences.append(input_data[i:i+window_size])
        labels.append(target_data[i+window_size])
    return np.array(sequences), np.array(labels)

input_data = np.load(input_features_path)
target_data = np.load(target_features_path)
print("Número de características objetivo:", target_data.shape[1])

# División del dataset basada en tiempo
test_size = 30
train_input = input_data[:-test_size]
test_input = input_data[-test_size:]
train_target = target_data[:-test_size]
test_target = target_data[-test_size:]

# Crear secuencias de entrenamiento y prueba
window_size = 5
X_train, y_train = create_sequences(train_input, train_target, window_size)
X_test, y_test = create_sequences(test_input, test_target, window_size)

#GRU
model = Sequential([
    GRU(128, input_shape=(window_size, X_train.shape[2]), activation='relu', return_sequences=True,
        kernel_regularizer=L1L2(l1=1e-5, l2=1e-4)),
    BatchNormalization(),
    Dropout(0.3),
    GRU(64, activation='relu', return_sequences=True,
        kernel_regularizer=L1L2(l1=1e-5, l2=1e-4)),
    BatchNormalization(),
    Dropout(0.3),
    GRU(32, activation='relu',
        kernel_regularizer=L1L2(l1=1e-5, l2=1e-4)),
    Dropout(0.3),
    Dense(y_train.shape[1], activation='linear')
])

model.compile(optimizer=Adam(learning_rate=0.001), 
            loss='mean_squared_error', 
            metrics=[MeanSquaredError(), MeanAbsoluteError(), MeanSquaredLogarithmicError(), RootMeanSquaredError()])


# Adicion del "EarlyStopping"
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test), 
                    callbacks=[early_stopping], verbose=1)

best_epoch = early_stopping.stopped_epoch - early_stopping.patience
print(f"El entrenamiento se detuvo en el epoch: {best_epoch + 1}")
print("Métricas en el punto de detención del Early Stopping:")

print(history.history)

current_date = datetime.now().strftime("%Y%m%d")
model.save(model_save_path) # se guarda el modelo con la fecha en que fue entrenado
print(f"Modelo guardado como {model_save_path}")

metrics = {
    "MeanSquaredError": history.history['mean_squared_error'][best_epoch],
    "MeanAbsoluteError": history.history['mean_absolute_error'][best_epoch],
    "MeanSquaredLogarithmicError": history.history['mean_squared_logarithmic_error'][best_epoch],
    "RootMeanSquaredError": history.history['root_mean_squared_error'][best_epoch]
}

with open(metrics_save_path, 'w') as file:
    json.dump(metrics, file, indent=4)
print(f"Métricas guardadas en {metrics_save_path}")


print("Métricas guardadas en metrics.json")
print("Historial de entrenamiento:")
for metric in history.history.keys():
    print(f"{metric}: {history.history[metric][best_epoch]}")
