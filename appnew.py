import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Cargar el modelo entrenado
with open('modelo_optimizado_Grupo3.pkl', 'rb') as file:
    modelo = pickle.load(file)

# Definir la interfaz de usuario en Streamlit
st.title('Predicción de Precios de Laptops')

# Controles de entrada para las características
Ram = st.number_input('Ram (GB)', min_value=1, max_value=64, value=8)
screen_width = st.number_input('Ancho de Pantalla', min_value=800, max_value=4000, value=1920)
screen_height = st.number_input('Alto de Pantalla', min_value=600, max_value=3000, value=1080)
TypeName_Gaming = st.selectbox('¿Es Gaming?', ['No', 'Sí'])
Cpu = st.number_input('Cpu', min_value=74, max_value=116, value=74)
TypeName_Ultrabook = st.selectbox('¿Es Ultrabook?', ['No', 'Sí'])

# Convertir entradas a formato numérico
TypeName_Gaming = 1 if TypeName_Gaming == 'Sí' else 0
TypeName_Ultrabook = 1 if TypeName_Ultrabook == 'Sí' else 0

# Botón para realizar predicción
if st.button('Predecir Precio'):
    # Crear DataFrame con las entradas
    input_data = pd.DataFrame([[Ram, screen_width, screen_height, TypeName_Gaming, Cpu, TypeName_Ultrabook ]],
                    columns=['Ram', 'screen_width', 'screen_height', 'TypeName_Gaming', 'Cpu', 'TypeName_Ultrabook'])

    # Estandarización de las características
    scaler=StandardScaler()
    X_sc = scaler.fit_transform(X)

    # Realizar predicción
    prediction = modelo.predict(X_sc)

    # Mostrar predicción
    st.write(f'Precio predecido: {prediction[0]:.2f} euros')


