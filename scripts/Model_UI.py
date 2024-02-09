import streamlit as st
import requests
import pandas as pd
import os

st.set_page_config(page_title="Predicciones Climáticas", layout="wide")

st.title("Predicciones Climáticas 🌦️")
st.subheader("Modelo GRU entrenado con series de tiempo climáticas 📈")

st.write("""
        Este proyecto utiliza un modelo GRU (Gated Recurrent Unit) entrenado con datos históricos de series de tiempo climáticas para predecir condiciones climáticas futuras, como temperatura, presión atmosférica, humedad, etc. 🌡️💨🌧️. 
        
        El dataset utilizado, contiene datos desde Agosto 2023 y se actualiza diariamente, el modelo GRU se reentrena automaticamente cada 7 dias y se elige el mejor modelo para continuar realizando las predicciones.
        
        La aplicación permite a los usuarios obtener predicciones climáticas específicas para una ubicación y fecha/hora determinadas, basándose en la potente arquitectura de redes neuronales recurrentes para procesar secuencias de datos temporales.
        
        Dataset link [here](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository)
        """)

ubicacion = st.text_input("Ubicación (Ciudad) 🏙️", "Mexico City")
pais = st.text_input("País 🌍", "Mexico")
st.write("Fecha y Hora ⏰")
fecha_hora = st.text_input("AAAA-MM-DD HH:MM", "2024-02-01 19:30")


if st.button("Realizar Predicción"):
    pred_key = f"{ubicacion}-{pais}-{fecha_hora}"
    respuesta = requests.get(f"http://localhost:8000/predict/?ubicacion_usuario={ubicacion}&pais_usuario={pais}&fecha_hora_usuario={fecha_hora}")

    if respuesta.status_code == 200:
        datos_prediccion = respuesta.json()
        df_prediccion = pd.DataFrame(list(datos_prediccion.items()), columns=['Variable', 'Valor'])
        st.session_state['last_response'] = df_prediccion
        st.session_state['last_pred_key'] = pred_key
        st.markdown(f"**Predicción para {ubicacion}, {pais} en {fecha_hora}:**")
        st.table(df_prediccion)
    else:
        st.error("Error en la obtención de la predicción. Por favor, inténtalo de nuevo.")


if 'last_response' in st.session_state:
    #st.markdown(f"**Última predicción para {ubicacion}, {pais} en {fecha_hora}:**")
    # st.table(st.session_state['last_response'])

    # Solo mostrar la opción de feedback si hay una predicción
    feedback_options = ('👍 Positivo', '👎 Negativo')
    feedback = st.radio("Califica mi predicción", options=feedback_options, key="feedback_radio")

    # Botón para enviar feedback
    if st.button("Enviar Feedback", key="submit_feedback"):
        pred_key = st.session_state.get('last_pred_key')
        if pred_key and not st.session_state.get(f'feedback_sent_{pred_key}', False):
            feedback_val = "positivo" if feedback == '👍 Positivo' else "negativo"
            feedback_file = "../model_data/feedback.csv"
            new_feedback = pd.DataFrame({
                'ubicacion': [ubicacion], 
                'pais': [pais], 
                'fecha_hora': [fecha_hora], 
                'feedback': [feedback_val], 
                'calificacion': [1 if feedback_val == "positivo" else 0]
            })

            if os.path.exists(feedback_file):
                df_feedback = pd.read_csv(feedback_file)
            else:
                df_feedback = pd.DataFrame(columns=['ubicacion', 'pais', 'fecha_hora', 'feedback', 'calificacion'])

            df_feedback = pd.concat([df_feedback, new_feedback], ignore_index=True)
            df_feedback.to_csv(feedback_file, index=False)

            st.success("¡Gracias por tu feedback!")
            st.session_state[f'feedback_sent_{pred_key}'] = True  # Marcar feedback como enviado para esta predicción
        else:
            st.error("Ya has enviado feedback para esta predicción.")