# Predicción de Condiciones Climáticas

## Descripción
Este proyecto aplica técnicas de machine learning para predecir condiciones climáticas utilizando series de tiempo. 
Utiliza un dataset global actualizado diariamente para entrenar un modelo basado en Redes Neuronales Recurrentes (RNN), específicamente, unidades GRU (Gated Recurrent Unit), debido a su efectividad en el manejo de secuencias temporales.

El dataset se encuentra en Kaggle e incluye más de 40 características, como temperatura, velocidad y dirección del viento, presión, precipitación, humedad, visibilidad, y mediciones de calidad del aire, entre otros. Es un recurso valioso para analizar patrones climáticos globales, explorar tendencias climáticas y comprender las relaciones entre diferentes parámetros meteorológicos. Las características clave abarcan desde datos básicos de ubicación hasta índices de calidad del aire y fases lunares, ofreciendo un amplio espectro para análisis climáticos, predicciones meteorológicas, estudios de impacto ambiental, planificación turística y exploración de patrones geográficos.

[Aqui](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository) puedes consultar el dataset!!

El proyecto se construye sobre una arquitectura modular que integra diversos componentes para
automatizar la descarga diaria del dataset climático, el entrenamiento semanal del modelo GRU y la interacción
con los usuarios finales.

## Tecnologías Utilizadas
- Python
- FastAPI
- Streamlit
- TensorFlow/Keras
- Airflow

## Características
- **Interfaz Web:** Acceso a predicciones climáticas a través de una interfaz de usuario intuitiva.
- **Automatización:** Actualización diaria del dataset y reentrenamiento del modelo mediante DAG en Airflow.
- **Feedback de Usuarios:** Sección para recoger opiniones de los usuarios para poder realizar analisis posteriores.

**Airflow para Automatización:**
- Utilice Airflow para automatizar tareas críticas como la descarga diaria de datos climáticos y el
reentrenamiento semanal del modelo GRU. Esto asegura que el modelo siempre esté actualizado
con los últimos datos y optimizado para la precisión en las predicciones.

**API con FastAPI:**
- La API desarrollada en FastAPI actúa como el backend, procesando las solicitudes de predicción,
ejecutando el modelo de predicción climática y devolviendo los resultados a los usuarios.
La elección de FastAPI se debe a su alto rendimiento, facilidad de uso y soporte para operaciones
asincrónicas.

**Interfaz de Usuario con Streamlit:**
- La UI, creada con Streamlit, ofrece una experiencia interactiva y amigable para que los usuarios
realicen consultas de predicción climática basadas en la ubicación y la fecha/hora.
Streamlit fue seleccionado por su capacidad para desarrollar rápidamente interfaces ricas en datos
con mínima configuración.

**Escalabilidad y Mantenimiento:**
- Actualmente, el proyecto está configurado para una ejecución automatica con planes de revisión
continua de los modelos desplegados y sus métricas para asegurar la calidad de las predicciones.
Aunque no se han implementado planes específicos de escalabilidad, la arquitectura permite
futuras expansiones, como el despliegue en plataformas en la nube para manejar volúmenes
mayores de usuarios y datos.


## Diagrama del proyecto:
![image](https://github.com/SleepWalKer09/Climate_Prediction_App/assets/44912298/c2186f88-346d-43ff-bb98-36bbb3de7e71)


## Instalación y Uso
1. Para iniciar la aplicación, asegúrate de tener instaladas las dependencias necesarias.
   Recomiendo que para la instalacion de Airflow sigas la documentacion y que todo lo ejecutes desde WSL:

```bash
# Instalar dependencias (asegúrate de tener un entorno virtual activo)
pip install -r requirements. txt
```

Es posible que durante la configuracion de Airflow desde WSL indique que existe un error de compatibilidad con las dependencias de flask, si es el caso, lo unico que tienes que hacer es:
```bash
## Bajar la version de la dependencia que esta ocacionando el error
pip install Flask-Session==0.4.0
```
  Solucion propuesta por los autores [aqui](https://github.com/apache/airflow/pull/36895)

2. Consultar, activar/desactivar DAG (Luego de instalar y configurar exitosamente Airflow):
   
```bash
## Iniciar el servidor web de Airflow
airflow webserver -p 8080
## Iniciar el scheduler de Airflow
airflow scheduler
```

3. Ejecutar la API y la UI:
```bash
# Iniciar el servidor FastAPI
uvicorn main:app --reload

# Ejecutar Streamlit UI
streamlit run app.py
```
