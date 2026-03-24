def estadisticas(df):
    numericas = df.select_dtypes(include=['number'])

    if numericas.empty:
        return "No hay columnas numéricas para calcular estadísticas descriptivas."

    return numericas.describe()

def correlacion(df):
    numericas = df.select_dtypes(include=['number'])

    if numericas.empty:
        return "No hay columnas numéricas para calcular correlación."

    return numericas.corr()

def resumen_categorico(df):
    resumen = {}

    columnas_categoricas = df.select_dtypes(include=['object']).columns

    for col in columnas_categoricas:
        resumen[col] = df[col].value_counts().head(5)

    return resumen

def valores_nulos(df):
    return df.isnull().sum()

def dimensiones(df):
    filas, columnas = df.shape
    return {
        "filas": filas,
        "columnas": columnas
    }