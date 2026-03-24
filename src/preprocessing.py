import pandas as pd

def limpiar_datos(df):
    df = df.drop_duplicates()
    return df

def manejar_nulos(df):
    columnas_numericas = df.select_dtypes(include=['number']).columns

    if len(columnas_numericas) > 0:
        df[columnas_numericas] = df[columnas_numericas].fillna(
            df[columnas_numericas].mean()
        )

    return df

def convertir_fechas(df):
    for col in df.columns:
        # solo intentar convertir si la columna es texto
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')

                # si más del 70% son fechas válidas, se deja como fecha
                if df[col].notnull().sum() / len(df) < 0.7:
                    df[col] = df[col].astype(str)

            except:
                pass

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