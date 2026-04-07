import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import cargar_datos
from src.preprocessing import limpiar_datos, manejar_nulos, convertir_fechas, detectar_tipos
from src.analysis import estadisticas, correlacion, resumen_categorico, valores_nulos, dimensiones
from src.ia import aplicar_clustering, detectar_outliers, generar_insights

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Proyecto IA - Análisis de Datos",
    page_icon="🤖",
    layout="wide"
)

# ─────────────────────────────────────────────
# ENCABEZADO
# ─────────────────────────────────────────────
st.title("🤖 Proyecto IA — Análisis Automatizado de Datos")
st.markdown("Carga un archivo **CSV o Excel** y la app analiza tus datos automáticamente.")
st.divider()

# ─────────────────────────────────────────────
# CARGA DE ARCHIVO
# ─────────────────────────────────────────────
archivo = st.file_uploader("📂 Sube un archivo CSV o Excel", type=["csv", "xlsx"])

if archivo:
    df = cargar_datos(archivo)

    if df is not None:

        # Preprocesamiento
        df = limpiar_datos(df)
        df = manejar_nulos(df)
        df = convertir_fechas(df)

        # Datos calculados
        tipos      = detectar_tipos(df)
        dims       = dimensiones(df)
        nulos      = valores_nulos(df)
        stats      = estadisticas(df)
        corr       = correlacion(df)
        resumen_cat = resumen_categorico(df)
        numericas  = df.select_dtypes(include=["number"]).columns.tolist()

        # IA
        n_clusters  = 3
        df_cluster  = aplicar_clustering(df, n_clusters=n_clusters)
        outliers    = detectar_outliers(df)
        insights    = generar_insights(df_cluster, outliers, n_clusters=n_clusters)

        # ─────────────────────────────────────────────
        # TABS DE NAVEGACIÓN
        # ─────────────────────────────────────────────
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Resumen",
            "📊 Gráficos",
            "🤖 IA & Clustering",
            "⚠️ Outliers",
            "💡 Insights"
        ])

        # ══════════════════════════════════════════════
        # TAB 1 — RESUMEN GENERAL
        # ══════════════════════════════════════════════
        with tab1:
            st.subheader("Vista previa del dataset")
            st.dataframe(df.head(10), use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("📌 Filas", dims["filas"])
            col2.metric("📌 Columnas", dims["columnas"])
            col3.metric("📌 Valores nulos", int(nulos.sum()))

            st.divider()

            col_a, col_b = st.columns(2)

            with col_a:
                st.subheader("🔢 Tipos de variables")
                st.json(tipos)

            with col_b:
                st.subheader("🕳️ Valores nulos por columna")
                st.dataframe(
                    nulos.reset_index().rename(columns={"index": "Columna", 0: "Nulos"}),
                    use_container_width=True
                )

            st.divider()
            st.subheader("📈 Estadísticas descriptivas")
            st.dataframe(stats, use_container_width=True)

            if isinstance(corr, pd.DataFrame):
                st.divider()
                st.subheader("🔗 Matriz de correlación (valores)")
                st.dataframe(corr.style.background_gradient(cmap="coolwarm"), use_container_width=True)

            if resumen_cat:
                st.divider()
                st.subheader("🏷️ Resumen categórico")
                for col, vals in resumen_cat.items():
                    with st.expander(f"Top 5 — {col}"):
                        st.dataframe(vals)

        # ══════════════════════════════════════════════
        # TAB 2 — GRÁFICOS
        # ══════════════════════════════════════════════
        with tab2:

            if len(numericas) == 0:
                st.warning("No hay columnas numéricas para graficar.")
            else:

                # ── Heatmap de correlación ──
                st.subheader("🌡️ Heatmap de Correlación")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(
                    df[numericas].corr(),
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm",
                    ax=ax,
                    linewidths=0.5
                )
                ax.set_title("Matriz de correlación")
                st.pyplot(fig)
                plt.close()

                st.divider()

                # ── Histogramas ──
                st.subheader("📊 Histogramas")
                col_hist = st.selectbox("Selecciona una variable:", numericas, key="hist")
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(df[col_hist].dropna(), kde=True, color="steelblue", ax=ax)
                ax.set_title(f"Distribución de {col_hist}")
                ax.set_xlabel(col_hist)
                st.pyplot(fig)
                plt.close()

                st.divider()

                # ── Boxplots ──
                st.subheader("📦 Boxplots — Detección visual de outliers")
                col_box = st.selectbox("Selecciona una variable:", numericas, key="box")
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.boxplot(x=df[col_box].dropna(), color="coral", ax=ax)
                ax.set_title(f"Boxplot de {col_box}")
                st.pyplot(fig)
                plt.close()

                st.divider()

                # ── Scatter plot ──
                st.subheader("🔵 Scatter Plot — Relación entre variables")
                if len(numericas) >= 2:
                    col_x = st.selectbox("Eje X:", numericas, index=0, key="scatter_x")
                    col_y = st.selectbox("Eje Y:", numericas, index=1, key="scatter_y")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.scatterplot(
                        data=df,
                        x=col_x,
                        y=col_y,
                        alpha=0.6,
                        color="mediumseagreen",
                        ax=ax
                    )
                    ax.set_title(f"{col_x} vs {col_y}")
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("Se necesitan al menos 2 columnas numéricas para el scatter plot.")

        # ══════════════════════════════════════════════
        # TAB 3 — IA & CLUSTERING
        # ══════════════════════════════════════════════
        with tab3:
            st.subheader("🤖 Clustering — K-Means")

            if len(numericas) < 2:
                st.warning("Se necesitan al menos 2 columnas numéricas para aplicar clustering.")
            else:
                n_clusters = st.slider("Número de grupos (clusters):", 2, 6, 3)
                df_cluster = aplicar_clustering(df, n_clusters=n_clusters)

                st.markdown("**Distribución de grupos:**")
                conteo = df_cluster["cluster"].value_counts().sort_index().reset_index()
                conteo.columns = ["Cluster", "Cantidad"]
                st.dataframe(conteo, use_container_width=True)

                # Scatter coloreado por cluster
                st.markdown("**Visualización de clusters:**")
                col_x2 = st.selectbox("Eje X:", numericas, index=0, key="cx")
                col_y2 = st.selectbox("Eje Y:", numericas, index=1, key="cy")

                fig, ax = plt.subplots(figsize=(8, 5))
                scatter = ax.scatter(
                    df_cluster[col_x2],
                    df_cluster[col_y2],
                    c=df_cluster["cluster"],
                    cmap="Set1",
                    alpha=0.6
                )
                plt.colorbar(scatter, ax=ax, label="Cluster")
                ax.set_xlabel(col_x2)
                ax.set_ylabel(col_y2)
                ax.set_title(f"Clusters — {col_x2} vs {col_y2}")
                st.pyplot(fig)
                plt.close()

                st.divider()
                st.subheader("📋 Dataset con columna de cluster")
                st.dataframe(df_cluster.head(20), use_container_width=True)

        # ══════════════════════════════════════════════
        # TAB 4 — OUTLIERS
        # ══════════════════════════════════════════════
        with tab4:
            st.subheader("⚠️ Detección de Valores Atípicos")

            outliers_df = pd.DataFrame(outliers).T.reset_index()
            outliers_df.columns = ["Variable", "Outliers (IQR)", "Outliers (Z-score)", "% Outliers (IQR)"]
            st.dataframe(outliers_df, use_container_width=True)

            st.divider()
            st.subheader("📦 Boxplots de todas las variables")

            cols_por_fila = 3
            filas = [numericas[i:i+cols_por_fila] for i in range(0, len(numericas), cols_por_fila)]

            for fila in filas:
                cols = st.columns(len(fila))
                for idx, col_name in enumerate(fila):
                    with cols[idx]:
                        fig, ax = plt.subplots(figsize=(4, 3))
                        sns.boxplot(y=df[col_name].dropna(), color="coral", ax=ax)
                        ax.set_title(col_name, fontsize=10)
                        st.pyplot(fig)
                        plt.close()

        # ══════════════════════════════════════════════
        # TAB 5 — INSIGHTS
        # ══════════════════════════════════════════════
        with tab5:
            st.subheader("💡 Insights Automáticos")
            st.markdown("Hallazgos generados automáticamente por el sistema de IA:")
            st.divider()

            for i, insight in enumerate(insights, 1):
                st.info(f"**{i}.** {insight}")

else:
    # Pantalla de bienvenida
    st.markdown("""
    ### 👈 Comienza subiendo un archivo

    La aplicación analizará automáticamente tus datos y generará:

    - 📋 **Resumen** — estadísticas descriptivas y tipos de variables
    - 📊 **Gráficos** — histogramas, boxplots, scatter y heatmap
    - 🤖 **Clustering** — agrupación automática con K-Means
    - ⚠️ **Outliers** — detección de valores atípicos
    - 💡 **Insights** — hallazgos automáticos en lenguaje natural

    **Formatos soportados:** CSV, Excel (.xlsx)
    """)