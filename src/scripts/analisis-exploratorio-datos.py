import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analisis-exploratorio-datos():

    %store -r df_QCX
    %store -r df_pilas

    df_QCX = df_QCX
    df_pilas = df_pilas

    # 0. Cambio tipo de datos
    df_pilas = df_pilas.astype({
        'Index': 'int64',
        ' Time': 'datetime64[ns]',
        ' Mound': 'object',
        ' Comment': 'object',
        'idPila': 'int64'
        })

    # Reemplazar valores no numéricos (como espacios en blanco) con NaN
    cols_to_convert = [
         "LSF"  [CurrentAnalysis.Dry basis]', ' "LSF"  [Rolling.Analysis1.Dry basis]', ' "SM"  [Rolling.Analysis1.Dry basis]',
        ' "IM"  [Rolling.Analysis1.Dry basis]', ' "Tph"  [Rolling.Analysis1.Dry basis]', ' "IM"  [CurrentProduct.Dry basis]',
        ' "LSF"  [CurrentProduct.Dry basis]', ' "SM"  [CurrentProduct.Dry basis]', ' "Tph"  [CurrentProduct.Dry basis]',
        ' "CaO"  [CurrentProduct.Dry basis]', ' "CaO"  [Rolling.Analysis1.Dry basis]', ' "MgO"  [CurrentProduct.Dry basis]',
        ' "MgO"  [Rolling.Analysis1.Dry basis]', ' "Al2O3"  [CurrentProduct.Dry basis]', ' "Fe2O3"  [CurrentProduct.Dry basis]',
        ' "Al2O3"  [Rolling.Analysis1.Dry basis]', ' "Fe2O3"  [Rolling.Analysis1.Dry basis]', ' "SiO2"  [CurrentProduct.Dry basis]',
        ' "SiO2"  [Rolling.Analysis1.Dry basis]', ' "SM"  [CurrentAnalysis.Dry basis]', ' "CaO"  [CurrentAnalysis.Dry basis]',
        ' "MgO"  [CurrentAnalysis.Dry basis]', ' "IM"  [CurrentAnalysis.Dry basis]', ' "Fe2O3"  [CurrentAnalysis.Dry basis]',
        ' "Al2O3"  [CurrentAnalysis.Dry basis]', ' "SiO2"  [CurrentAnalysis.Dry basis]', ' "Tph"  [CurrentAnalysis.Dry basis]',
        ' "Tons"  [CurrentProduct.Dry basis]'
    ]

    # Reemplazar valores no numéricos o espacios en blanco con NaN en las columnas especificadas
    df_pilas[cols_to_convert] = df_pilas[cols_to_convert].replace(r'^\s*$', np.nan, regex=True)

    # Intentar nuevamente convertir las columnas a float64  
    df_pilas[cols_to_convert] = df_pilas[cols_to_convert].astype('float64')


    # Lista de columnas a procesar
    cols_to_convert2 = [
        'FCAO', 'SiO2', 'Al2O3',
        'Fe2O3', 'CaO', 'MgO', 'SO3', 'K2O', 'Na2O', 'C3S', 'C2S', 'C3A',
        'C4AF', 'A/S', 'idPila'
    ]

    # Reemplazar cualquier valor que no sea numérico por NaN en las columnas especificadas
    df_QCX[cols_to_convert2] = df_QCX[cols_to_convert2].applymap(lambda x: np.nan if not str(x).replace('.', '', 1).isdigit() else x)


    # 1. Análisis Exploratorio de Datos (EDA) df_pilas

    # Mostrar información básica
    print(f"El DataFrame tiene {df_pilas.shape[0]} filas y {df_pilas.shape[1]} columnas.")
    print("\nNombres de las columnas:")
    df_pilas.columns

    # Mostrar las primeras filas del DataFrame df_pilas
    print("\nPrimeras filas del DataFrame:")
    df_pilas.head()

    # 1. Tipos de datos de las columnas
    print("\nTipos de datos de las columnas:")
    print(df_pilas.dtypes)

    # 2. Valores nulos por columna
    print("\nValores nulos por columna:")
    df_pilas.isnull().mean()
    
    # 3. Estadísticas descriptivas de las variables numéricas
    print("\nResumen estadístico:")
    df_pilas.describe()

    # 4. Visualización de distribuciones de variables numéricas
    num_columns = df_pilas.select_dtypes(include=[np.number]).columns
    for col in num_columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df_pilas[col], kde=True, bins=30)
        plt.title(f'Distribución de {col}')
        plt.show()


    plt.figure(figsize=(12, 8))
    sns.heatmap(df_pilas.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Mapa de calor de la correlación entre variables numéricas')
    plt.show()

    # 2. Análisis Exploratorio de Datos (EDA) df_QCX

    # Mostrar información básica
    print(f"El DataFrame tiene {df_QCX.shape[0]} filas y {df_QCX.shape[1]} columnas.")
    print("\nNombres de las columnas:")
    df_QCX.columns

    # Mostrar las primeras filas del DataFrame df_QCX
    print("\nPrimeras filas del DataFrame:")
    df_QCX.head()

    # 1. Tipos de datos de las columnas
    print("\nTipos de datos de las columnas:")
    df_QCX.dtypes

    # 2. Valores nulos por columna
    print("\nValores nulos por columna:")
    df_QCX.isnull().mean()

    # 3. Estadísticas descriptivas de las variables numéricas
    print("\nResumen estadístico:")
    df_QCX.describe()


    # 4. Visualización de distribuciones de variables numéricas
    num_columns = df_QCX.select_dtypes(include=[np.number]).columns
    for col in num_columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df_QCX[col], kde=True, bins=30)
        plt.title(f'Distribución de {col}')
        plt.show()

    plt.figure(figsize=(12, 8))
    sns.heatmap(df_QCX.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Mapa de calor de la correlación entre variables numéricas')
    plt.show()


    %store df_QCX

    %store df_pilas
