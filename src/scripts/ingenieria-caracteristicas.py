import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def ingenieria-carcteristicas():
    %store -r df_QCX
    %store -r df_pilas

    df_QCX = df_QCX
    df_pilas = df_pilas

    # Primero vamos a eliminar las columnas que no necesitamos en el df_pilas
    df_pilas.columns

    # Lista de columnas a eliminar
    columns_to_drop = [
        'Index', ' Time', ' Period', ' Seconds since', ' Mound', ' Comment', ' "Tph"  [CurrentProduct.Dry basis]',
        ' "Tph"  [CurrentAnalysis.Dry basis]', ' "Tons"  [CurrentProduct.Dry basis]', ' "MgO"  [CurrentAnalysis.Dry basis]'
    ]

    # Eliminar las columnas del DataFrame
    df_pilas = df_pilas.drop(columns=columns_to_drop)

    # Luego filtramos las fechas y los id_eq que vamos a ocupar en el df_QCX

    df_QCX = df_QCX[(df_QCX['fecha'] >= '2023-01-01')]
    df_QCX = df_QCX[df_QCX['id_eq'].isin(['Man_CLK461', 'Man_CLK462', 'Aus_CLK463'])]
    
    # 0. Limpieza de valores nulos

    df_pilas.isnull().mean()*100
    df_pilas.shape
    df_pilas = df_pilas.dropna()
    df_pilas.shape
    df_QCX.isnull().mean()*100
    df_QCX.shape
    df_QCX = df_QCX.dropna()
    df_QCX.shape
    
    # 1. Creación de dataframe combinado entre df_pilas y df_QCX

    resumen_pilas = df_pilas.groupby('idPila').agg({
        ' "CaO"  [CurrentProduct.Dry basis]': 'last',
        ' "SiO2"  [CurrentProduct.Dry basis]': 'last',
        ' "Al2O3"  [CurrentProduct.Dry basis]': 'last',
        ' "Fe2O3"  [CurrentProduct.Dry basis]': 'last',
        ' "MgO"  [CurrentProduct.Dry basis]': 'last',
        ' "CaO"  [Rolling.Analysis1.Dry basis]': 'std',
        ' "SiO2"  [Rolling.Analysis1.Dry basis]': 'std',
        ' "Al2O3"  [Rolling.Analysis1.Dry basis]': 'std',
        ' "Fe2O3"  [Rolling.Analysis1.Dry basis]': 'std',
        ' "MgO"  [Rolling.Analysis1.Dry basis]': 'std'
    }).reset_index()

    resumen_pilas

    # Hacer una combinación tipo inner join en la columna 'idPila'
    dataset = pd.merge(df_QCX, resumen_pilas, on='idPila', how='inner')
    dataset.drop(['fecha', 'fecha-hora','Muestra #', 'idPila'], axis=1, inplace=True)

    # 2. Tratamiento de outliers
    # Definir las columnas numéricas
    numeric_columns = dataset.select_dtypes(include='float64').columns

    # Generar gráficos separados para cada variable
    for column in numeric_columns:
    # Crear figura y ejes
        fig, (ax_box, ax_kde) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    
    # Gráfico de cajas (boxplot)
        sns.boxplot(data=dataset[column], ax=ax_box, color='skyblue')
        ax_box.set_title(f'Boxplot de {column}')
    
    # Gráfico de densidad (kdeplot)
        sns.kdeplot(data=dataset[column], ax=ax_kde, color='orange', fill=True)
        ax_kde.set_title(f'Densidad de {column}')

    # Calcular los cuartiles y el rango intercuartílico (IQR)
    Q1 = dataset[numeric_columns].quantile(0.25)
    Q3 = dataset[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1

    # Definir los límites para los outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Remover los outliers
    for col in numeric_columns:
        dataset = dataset[(dataset[col] >= lower_bound[col]) & (dataset[col] <= upper_bound[col])]

    # Generar gráficos separados para cada variable
    for column in numeric_columns:
        # Crear figura y ejes
        fig, (ax_box, ax_kde) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    
        # Gráfico de cajas (boxplot)
        sns.boxplot(data=dataset[column], ax=ax_box, color='skyblue')
        ax_box.set_title(f'Boxplot de {column}')
    
    # Gráfico de densidad (kdeplot)
        sns.kdeplot(data=dataset[column], ax=ax_kde, color='orange', fill=True)
        ax_kde.set_title(f'Densidad de {column}')
    
    # 3. Codificación de variables categóricas

    dataset.columns

    from sklearn.preprocessing import LabelEncoder

         Crear una instancia de LabelEncoder
        label_encoder = LabelEncoder()

        # Aplicar la codificación a la columna 'id_eq'
        dataset['id_eq'] = label_encoder.fit_transform(dataset['id_eq'])

    dataset.to_csv('../data/processed/features_for_model.csv')


