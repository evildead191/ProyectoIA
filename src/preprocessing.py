import pandas as pd

def limpiar_datos(df):
    return df.drop_duplicates()

def manejar_nulos(df):
    columnas_numericas = df.select_dtypes(include=['number']).columns
    if len(columnas_numericas) > 0:
        df[columnas_numericas] = df[columnas_numericas].fillna(df[columnas_numericas].mean())
    return df

def detectar_tipos(df):
    numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categoricas = df.select_dtypes(include=['object']).columns.tolist()
    fechas = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()

    return {
        "numericas": numericas,
        "categoricas": categoricas,
        "fechas": fechas
    }