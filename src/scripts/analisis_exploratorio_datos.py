import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analisis_exploratorio_datos(df_pilas, df_qcx):
    """
    Realiza el análisis exploratorio de datos sobre los DataFrames proporcionados.

    Args:
        df_pilas (pd.DataFrame): DataFrame con datos históricos de pilas.
        df_qcx (pd.DataFrame): DataFrame con datos de calidad química.

    Returns:
        tuple: DataFrames procesados (df_pilas, df_qcx).
    """
    # 0. Cambio de tipo de datos
    df_pilas = df_pilas.astype({
        'Index': 'int64',
        'Time': 'datetime64[ns]',
        'Mound': 'object',
        'Comment': 'object',
        'idPila': 'int64'
    })

    # Reemplazar valores no numéricos o espacios en blanco con NaN
    cols_to_convert = [
        "LSF [CurrentAnalysis.Dry basis]",
        "LSF [Rolling.Analysis1.Dry basis]",
        "SM [Rolling.Analysis1.Dry basis]",
        "IM [Rolling.Analysis1.Dry basis]",
        "Tph [Rolling.Analysis1.Dry basis]",
        "IM [CurrentProduct.Dry basis]",
        "LSF [CurrentProduct.Dry basis]",
        "SM [CurrentProduct.Dry basis]",
        "Tph [CurrentProduct.Dry basis]",
        "CaO [CurrentProduct.Dry basis]",
        "CaO [Rolling.Analysis1.Dry basis]",
        "MgO [CurrentProduct.Dry basis]",
        "MgO [Rolling.Analysis1.Dry basis]",
        "Al2O3 [CurrentProduct.Dry basis]",
        "Fe2O3 [CurrentProduct.Dry basis]",
        "Al2O3 [Rolling.Analysis1.Dry basis]",
        "Fe2O3 [Rolling.Analysis1.Dry basis]",
        "SiO2 [CurrentProduct.Dry basis]",
        "SiO2 [Rolling.Analysis1.Dry basis]",
        "SM [CurrentAnalysis.Dry basis]",
        "CaO [CurrentAnalysis.Dry basis]",
        "MgO [CurrentAnalysis.Dry basis]",
        "IM [CurrentAnalysis.Dry basis]",
        "Fe2O3 [CurrentAnalysis.Dry basis]",
        "Al2O3 [CurrentAnalysis.Dry basis]",
        "SiO2 [CurrentAnalysis.Dry basis]",
        "Tph [CurrentAnalysis.Dry basis]",
        "Tons [CurrentProduct.Dry basis]"
    ]

    df_pilas[cols_to_convert] = df_pilas[cols_to_convert].replace(
        r'^\s*$', np.nan, regex=True
    ).astype('float64')

    # Reemplazar valores no numéricos en df_qcx
    cols_to_convert2 = [
        'FCAO', 'SiO2', 'Al2O3', 'Fe2O3', 'CaO', 'MgO', 'SO3',
        'K2O', 'Na2O', 'C3S', 'C2S', 'C3A', 'C4AF', 'A/S', 'idPila'
    ]

    df_qcx[cols_to_convert2] = df_qcx[cols_to_convert2].apply(pd.to_numeric, errors='coerce')

    # Análisis Exploratorio de df_pilas
    print(f"El DataFrame df_pilas tiene {df_pilas.shape[0]} filas y {df_pilas.shape[1]} columnas.")
    print("\nNombres de las columnas:", df_pilas.columns.tolist())
    print("\nPrimeras filas del DataFrame:\n", df_pilas.head())
    print("\nTipos de datos:\n", df_pilas.dtypes)
    print("\nValores nulos por columna:\n", df_pilas.isnull().mean())
    print("\nResumen estadístico:\n", df_pilas.describe())

    # Visualización
    num_columns = df_pilas.select_dtypes(include=[np.number]).columns
    for col in num_columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df_pilas[col], kde=True, bins=30)
        plt.title(f'Distribución de {col}')
        plt.show()

    plt.figure(figsize=(12, 8))
    sns.heatmap(df_pilas.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Mapa de calor de correlación - df_pilas')
    plt.show()

    # Análisis Exploratorio de df_qcx
    print(f"\nEl DataFrame df_qcx tiene {df_qcx.shape[0]} filas y {df_qcx.shape[1]} columnas.")
    print("\nNombres de las columnas:", df_qcx.columns.tolist())
    print("\nPrimeras filas del DataFrame:\n", df_qcx.head())
    print("\nTipos de datos:\n", df_qcx.dtypes)
    print("\nValores nulos por columna:\n", df_qcx.isnull().mean())
    print("\nResumen estadístico:\n", df_qcx.describe())

    # Visualización
    num_columns = df_qcx.select_dtypes(include=[np.number]).columns
    for col in num_columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df_qcx[col], kde=True, bins=30)
        plt.title(f'Distribución de {col}')
        plt.show()

    plt.figure(figsize=(12, 8))
    sns.heatmap(df_qcx.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Mapa de calor de la correlación entre variables numéricas')
    plt.show()

    return df_pilas, df_qcx
