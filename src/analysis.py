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

from src.data_loader import cargar_datos
from src.preprocessing import limpiar_datos, manejar_nulos, detectar_tipos
from src.analysis import estadisticas, correlacion

def pipeline_completo(file):
    df = cargar_datos(file)

    if df is None:
        return None

    df = limpiar_datos(df)
    df = manejar_nulos(df)
    tipos = detectar_tipos(df)

    stats = estadisticas(df)
    corr = correlacion(df)

    return df, tipos, stats, corr