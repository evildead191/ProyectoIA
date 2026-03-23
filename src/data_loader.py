import pandas as pd

def cargar_datos(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            raise ValueError("Formato no soportado")
        
        return df

    except Exception as e:
        print(f"Error al cargar datos: {e}")
        return None