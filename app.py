import streamlit as st
from src.data_loader import cargar_datos
from src.preprocessing import limpiar_datos, manejar_nulos, detectar_tipos
from src.analysis import estadisticas, correlacion

st.title("Proyecto IA - Análisis de Datos")

archivo = st.file_uploader("Sube un archivo CSV o Excel")

if archivo:
    df = cargar_datos(archivo)

    if df is not None:
        df = limpiar_datos(df)
        df = manejar_nulos(df)
        tipos = detectar_tipos(df)

        st.subheader("Vista previa")
        st.write(df.head())

        st.subheader("Tipos de datos")
        st.write(tipos)

        st.subheader("Estadísticas")
        st.write(estadisticas(df))

        st.subheader("Correlación")
        st.write(correlacion(df))