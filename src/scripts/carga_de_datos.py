import os
import pandas as pd

def carga_de_datos():
    """
    Carga y combina datos de archivos CSV de una carpeta específica
    y extrae información para crear un DataFrame combinado.
    """
    # Ruta de la carpeta donde están los archivos CSV
    folder_path = r"../data/raw/HistorialPilas"

    # Verificar si la carpeta existe
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"La carpeta {folder_path} no existe.")

    # Obtener la lista de todos los archivos CSV en la carpeta
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {folder_path}.")

    # Crear una lista para almacenar los DataFrames
    dataframes = []

    # Leer cada archivo CSV y agregarlo a la lista de DataFrames
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)
        dataframes.append(df)

    # Concatenar todos los DataFrames en uno solo
    df_pilas = pd.concat(dataframes, ignore_index=True)

    # Aplicar regex para extraer el primer número después de la palabra "Pila"
    if 'Mound' in df_pilas.columns:
        df_pilas['idPila'] = df_pilas['Mound'].str.extract(r'Pila (\d+)', expand=False)
    else:
        raise KeyError("La columna 'Mound' no existe en los datos cargados.")

    # Cargar el archivo Excel de datos de calidad
    calidad_path = "../data/raw/dataQCX.xlsx"
    if not os.path.exists(calidad_path):
        raise FileNotFoundError(f"El archivo {calidad_path} no existe.")

    df_QCX = pd.read_excel(calidad_path)

    # Retornar los DataFrames
    return df_pilas, df_QCX
