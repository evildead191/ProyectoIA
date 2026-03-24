import streamlit as st
from src.data_loader import cargar_datos
from src.preprocessing import limpiar_datos, manejar_nulos, convertir_fechas, detectar_tipos
from src.analysis import estadisticas, correlacion, resumen_categorico, valores_nulos, dimensiones

st.set_page_config(page_title="Proyecto IA - Análisis de Datos", layout="wide")

st.title("Proyecto IA - Análisis de Datos")
st.write("Carga un archivo CSV o Excel para analizar sus datos automáticamente.")

archivo = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx"])

if archivo:
    df = cargar_datos(archivo)

    if df is not None:
        df_original = df.copy()

        df = limpiar_datos(df)
        df = manejar_nulos(df)
        df = convertir_fechas(df)

        tipos = detectar_tipos(df)
        dims = dimensiones(df)
        nulos = valores_nulos(df)
        stats = estadisticas(df)
        corr = correlacion(df)
        resumen_cat = resumen_categorico(df)

        st.subheader("Vista previa")
        st.dataframe(df.head())

        st.subheader("Dimensiones del archivo")
        st.write(dims)

        st.subheader("Tipos de datos")
        st.json(tipos)

        st.subheader("Valores nulos por columna")
        st.write(nulos)

        st.subheader("Estadísticas descriptivas")
        st.write(stats)

        st.subheader("Correlación")
        st.write(corr)

        st.subheader("Resumen categórico")
        if len(resumen_cat) == 0:
            st.write("No hay columnas categóricas para resumir.")
        else:
            for columna, valores in resumen_cat.items():
                st.write(f"Top 5 valores en: {columna}")
                st.write(valores)