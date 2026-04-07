import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import cargar_datos
from src.preprocessing import limpiar_datos, manejar_nulos, convertir_fechas, detectar_tipos
from src.analysis import estadisticas, correlacion, resumen_categorico, valores_nulos, dimensiones
from src.ia import aplicar_clustering, detectar_outliers, generar_insights

st.set_page_config(
    page_title="Proyecto IA - Análisis de Datos",
    page_icon="🤖",
    layout="wide"
)

# SIDEBAR
st.sidebar.title("⚙️ Opciones")
mostrar_raw = st.sidebar.checkbox("Mostrar dataset completo")

st.title("🤖 Proyecto IA — Análisis Automatizado de Datos")
st.markdown("Carga un archivo **CSV o Excel** y la app analiza tus datos automáticamente.")
st.divider()

archivo = st.file_uploader("📂 Sube un archivo CSV o Excel", type=["csv", "xlsx"])

if archivo:
    try:
        with st.spinner("Procesando datos..."):
            df = cargar_datos(archivo)

            df = limpiar_datos(df)
            df = manejar_nulos(df)
            df = convertir_fechas(df)

            tipos = detectar_tipos(df)
            dims = dimensiones(df)
            nulos = valores_nulos(df)
            stats = estadisticas(df)
            corr = correlacion(df)
            resumen_cat = resumen_categorico(df)
            numericas = df.select_dtypes(include=["number"]).columns.tolist()

            df_cluster = aplicar_clustering(df, n_clusters=3)
            outliers = detectar_outliers(df)
            insights = generar_insights(df_cluster, outliers, n_clusters=3)

        if mostrar_raw:
            st.subheader("📄 Dataset completo")
            st.dataframe(df, use_container_width=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Resumen",
            "📊 Gráficos",
            "🤖 IA & Clustering",
            "⚠️ Outliers",
            "💡 Insights"
        ])

        # RESUMEN
        with tab1:
            st.subheader("Vista previa del dataset")
            st.dataframe(df.head(10), use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Filas", dims["filas"])
            col2.metric("Columnas", dims["columnas"])
            col3.metric("Valores nulos", int(nulos.sum()))

            st.divider()

            st.subheader("Correlaciones principales")
            if len(numericas) > 0:
                corr_matrix = df[numericas].corr().abs()
                top_corr = corr_matrix.unstack().sort_values(ascending=False)
                st.dataframe(top_corr.head(10))

        # GRAFICOS
        with tab2:
            if len(numericas) > 0:

                st.subheader("Heatmap de correlación")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(df[numericas].corr(), cmap="coolwarm", annot=False, square=True)
                st.pyplot(fig)
                plt.close()

                st.subheader("Histogramas")
                col_hist = st.selectbox("Variable", numericas)
                fig, ax = plt.subplots()
                sns.histplot(df[col_hist], kde=True, bins=30, color="skyblue")
                st.pyplot(fig)
                plt.close()

                st.subheader("Boxplot")
                fig, ax = plt.subplots()
                sns.boxplot(x=df[col_hist], color="lightcoral")
                sns.stripplot(x=df[col_hist], color="black", alpha=0.3)
                st.pyplot(fig)
                plt.close()

                if len(numericas) >= 2:
                    col_x = st.selectbox("X", numericas, key="x")
                    col_y = st.selectbox("Y", numericas, key="y")
                    fig, ax = plt.subplots()
                    sns.scatterplot(data=df, x=col_x, y=col_y, hue=col_x, size=col_y, palette="viridis")
                    st.pyplot(fig)
                    plt.close()

        # CLUSTERING
        with tab3:
            if len(numericas) >= 2:
                n_clusters = st.slider("Clusters", 2, 6, 3)
                df_cluster = aplicar_clustering(df, n_clusters=n_clusters)

                st.subheader("Distribución de clusters")
                fig, ax = plt.subplots()
                sns.countplot(x="cluster", data=df_cluster)
                st.pyplot(fig)
                plt.close()

                col_x = st.selectbox("X", numericas, key="cx")
                col_y = st.selectbox("Y", numericas, key="cy")

                fig, ax = plt.subplots()
                sns.scatterplot(data=df_cluster, x=col_x, y=col_y, hue="cluster", palette="Set1")
                st.pyplot(fig)
                plt.close()

        # OUTLIERS
        with tab4:
            outliers_df = pd.DataFrame(outliers).T.reset_index()
            outliers_df.columns = ["Variable", "IQR", "Z-score", "%"]
            st.dataframe(outliers_df)

        # INSIGHTS
        with tab5:
            for i, insight in enumerate(insights, 1):
                st.success(f"{i}. {insight}")

    except Exception as e:
        st.error("Error procesando el archivo")
        st.exception(e)

else:
    st.info("Sube un archivo para comenzar")