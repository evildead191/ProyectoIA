"""
ia.py - Módulo de Inteligencia Artificial
Persona 3: IA / Análisis inteligente

Funciones:
    - aplicar_clustering(): K-Means sobre columnas numéricas
    - detectar_outliers(): IQR y Z-score
    - generar_insights(): texto automático con hallazgos
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# ─────────────────────────────────────────────
# 1. CLUSTERING — K-Means
# ─────────────────────────────────────────────

def aplicar_clustering(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """
    Aplica K-Means sobre las columnas numéricas del dataframe.

    Parámetros:
        df         : DataFrame original
        n_clusters : Número de grupos (default: 3)

    Retorna:
        DataFrame original con una columna nueva 'cluster' (0, 1, 2, ...)
        o el DataFrame original si no hay columnas numéricas suficientes.
    """
    numericas = df.select_dtypes(include=["number"]).columns.tolist()

    if len(numericas) < 2:
        return df  # No hay suficientes columnas para agrupar

    datos = df[numericas].copy()

    # Imputar nulos con la media antes de escalar
    imputer = SimpleImputer(strategy="mean")
    datos_imputados = imputer.fit_transform(datos)

    # Escalar los datos (importante para K-Means)
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(datos_imputados)

    # Aplicar K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["cluster"] = kmeans.fit_predict(datos_escalados)

    return df


# ─────────────────────────────────────────────
# 2. DETECCIÓN DE OUTLIERS — IQR y Z-score
# ─────────────────────────────────────────────

def detectar_outliers(df: pd.DataFrame) -> dict:
    """
    Detecta valores atípicos usando IQR y Z-score en columnas numéricas.

    Parámetros:
        df : DataFrame original

    Retorna:
        Diccionario con resultados por columna:
        {
            "columna": {
                "iqr_outliers": int,      # cantidad de outliers por IQR
                "zscore_outliers": int,   # cantidad de outliers por Z-score
                "porcentaje_iqr": float,  # % del total
            },
            ...
        }
    """
    numericas = df.select_dtypes(include=["number"]).columns.tolist()
    resultados = {}

    for col in numericas:
        serie = df[col].dropna()

        # ── Método IQR ──
        Q1 = serie.quantile(0.25)
        Q3 = serie.quantile(0.75)
        IQR = Q3 - Q1
        limite_inf = Q1 - 1.5 * IQR
        limite_sup = Q3 + 1.5 * IQR
        outliers_iqr = ((serie < limite_inf) | (serie > limite_sup)).sum()

        # ── Método Z-score ──
        z_scores = np.abs((serie - serie.mean()) / serie.std())
        outliers_zscore = (z_scores > 3).sum()

        porcentaje = round((outliers_iqr / len(serie)) * 100, 2) if len(serie) > 0 else 0

        resultados[col] = {
            "iqr_outliers": int(outliers_iqr),
            "zscore_outliers": int(outliers_zscore),
            "porcentaje_iqr": porcentaje,
        }

    return resultados


# ─────────────────────────────────────────────
# 3. INSIGHTS AUTOMÁTICOS
# ─────────────────────────────────────────────

def generar_insights(df: pd.DataFrame, outliers: dict, n_clusters: int = 3) -> list:
    """
    Genera una lista de insights automáticos en texto legible.

    Parámetros:
        df         : DataFrame (con columna 'cluster' si ya se aplicó clustering)
        outliers   : Resultado de detectar_outliers()
        n_clusters : Número de clusters usados

    Retorna:
        Lista de strings con los hallazgos del análisis.

    Ejemplo de salida:
        [
          "Se identificaron 3 grupos principales en los datos.",
          "Se detectaron valores atípicos en 'alcohol' (8.5% de los registros).",
          "Existe una fuerte correlación entre 'density' y 'residual sugar' (r=0.84).",
        ]
    """
    insights = []
    numericas = df.select_dtypes(include=["number"]).columns.tolist()

    # ── Insight: Clustering ──
    if "cluster" in df.columns:
        conteo = df["cluster"].value_counts().sort_index()
        insights.append(
            f"Se identificaron {n_clusters} grupos principales en los datos. "
            + ", ".join([f"Grupo {i}: {v} registros" for i, v in conteo.items()]) + "."
        )
    else:
        insights.append(
            f"No se pudo aplicar clustering (se requieren al menos 2 columnas numéricas)."
        )

    # ── Insight: Outliers ──
    columnas_con_outliers = [
        col for col, res in outliers.items() if res["porcentaje_iqr"] > 0
    ]
    if columnas_con_outliers:
        for col in columnas_con_outliers:
            pct = outliers[col]["porcentaje_iqr"]
            n = outliers[col]["iqr_outliers"]
            if pct > 0:
                insights.append(
                    f"Se detectaron {n} valores atípicos en '{col}' "
                    f"({pct}% de los registros)."
                )
    else:
        insights.append("No se detectaron valores atípicos significativos en el dataset.")

    # ── Insight: Correlaciones fuertes ──
    if len(numericas) >= 2:
        cols_sin_cluster = [c for c in numericas if c != "cluster"]
        if len(cols_sin_cluster) >= 2:
            corr_matrix = df[cols_sin_cluster].corr().abs()
            # Obtener pares únicos (sin diagonal)
            pares = []
            for i in range(len(cols_sin_cluster)):
                for j in range(i + 1, len(cols_sin_cluster)):
                    val = corr_matrix.iloc[i, j]
                    pares.append((cols_sin_cluster[i], cols_sin_cluster[j], val))

            # Ordenar por correlación descendente
            pares.sort(key=lambda x: x[2], reverse=True)

            # Reportar las 3 correlaciones más fuertes (si r > 0.5)
            fuertes = [(a, b, r) for a, b, r in pares if r >= 0.5][:3]
            if fuertes:
                for a, b, r in fuertes:
                    nivel = "muy fuerte" if r >= 0.8 else "fuerte"
                    insights.append(
                        f"Existe una correlación {nivel} entre '{a}' y '{b}' (r={round(r, 2)})."
                    )
            else:
                insights.append("No se encontraron correlaciones fuertes entre las variables numéricas.")

    # ── Insight: Tamaño del dataset ──
    filas, cols = df.shape
    insights.append(
        f"El dataset contiene {filas} registros y {cols} variables analizadas."
    )

    # ── Insight: Nulos ──
    total_nulos = df.isnull().sum().sum()
    if total_nulos > 0:
        col_mas_nulos = df.isnull().sum().idxmax()
        insights.append(
            f"Se encontraron {total_nulos} valores nulos en total. "
            f"La columna con más nulos es '{col_mas_nulos}'."
        )
    else:
        insights.append("El dataset no contiene valores nulos.")

    return insights