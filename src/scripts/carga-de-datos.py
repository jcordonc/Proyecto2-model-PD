import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler

def carga-de-datos():

    """
    Carga y combina datos de archivos CSV de una carpeta específica
    y extrae información para crear un DataFrame combinado.
    """
    # Primero, vamos a importar los datos desde los archivos CSV que se encuentran en la carpeta historial pilas. 
    # Necesitamos unir un total de 186 archivos en un solo conjunto de datos.
    # Ruta de la carpeta donde están los archivos CSV
    folder_path = r"../data/raw/HistorialPilas"

    # Obtener la lista de todos los archivos CSV en la carpeta
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    # Crear una lista para almacenar los DataFrames
    dataframes = []

    # Leer cada archivo CSV y agregarlo a la lista de DataFrames
    for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    df = pd.read_csv(file_path)
    dataframes.append(df)

    # Concatenar todos los DataFrames en uno solo
    df_pilas = pd.concat(dataframes, ignore_index=True)

    # Usamos una expresión regular (regex) para extraer únicamente el número de pila 
    # de los datos y lo añadimos como una nueva columna llamada idPila.

    # Aplicar regex para extraer el primer número después de la palabra "Pila"
    df_pilas['idPila'] = df_pilas['Mound'].str.extract(r'Pila (\d+)', expand=False)


    # Ahora cargamos el archivo excel de los datos de calidad
    df_QCX = pd.read_excel("../data/raw/dataQCX.xlsx")


    %store df_QCX

    %store df_pilas
