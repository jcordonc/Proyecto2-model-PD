
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


def ingenieria_caracteristicas(df_QCX, df_pilas):
    """
    Realiza la ingeniería de características para combinar datos de análisis químico y pilas,
    eliminando valores nulos, tratando outliers y codificando variables categóricas.

    Args:
        df_QCX (pd.DataFrame): DataFrame con datos de análisis químico (QCX).
        df_pilas (pd.DataFrame): DataFrame con datos de pilas.

    Returns:
        None: Guarda el DataFrame procesado en un archivo CSV.
    """
    # Lista de columnas a eliminar
    columns_to_drop = [
        'Index', ' Time', ' Period', ' Seconds since', ' Mound', ' Comment',
        ' "Tph"  [CurrentProduct.Dry basis]', ' "Tph"  [CurrentAnalysis.Dry basis]',
        ' "Tons"  [CurrentProduct.Dry basis]', ' "MgO"  [CurrentAnalysis.Dry basis]'
    ]

    # Eliminar columnas irrelevantes
    df_pilas = df_pilas.drop(columns=columns_to_drop)

    # Filtrar fechas y equipos en df_QCX
    df_QCX = df_QCX[df_QCX['fecha'] >= '2023-01-01']
    df_QCX = df_QCX[df_QCX['id_eq'].isin(['Man_CLK461', 'Man_CLK462', 'Aus_CLK463'])]

    # Eliminar valores nulos
    df_pilas = df_pilas.dropna()
    df_QCX = df_QCX.dropna()

    # Crear resumen de pilas
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
        ' "MgO"  [Rolling.Analysis1.Dry basis]': 'std',
    }).reset_index()

    # Combinar los DataFrames
    dataset = pd.merge(df_QCX, resumen_pilas, on='idPila', how='inner')
    dataset.drop(['fecha', 'fecha-hora', 'Muestra #', 'idPila'], axis=1, inplace=True)

    # Tratamiento de outliers
    numeric_columns = dataset.select_dtypes(include='float64').columns
    Q1 = dataset[numeric_columns].quantile(0.25)
    Q3 = dataset[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1

    # Definir límites para outliers y filtrar datos
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    for col in numeric_columns:
        dataset = dataset[(dataset[col] >= lower_bound[col]) & (dataset[col] <= upper_bound[col])]

     # Generar gráficos de outliers (opcional, comentar si no se necesitan)
    for column in numeric_columns:
        fig, (ax_box, ax_kde) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        sns.boxplot(data=dataset[column], ax=ax_box, color='skyblue')
        ax_box.set_title(f'Boxplot de {column}')
        sns.kdeplot(data=dataset[column], ax=ax_kde, color='orange', fill=True)
        ax_kde.set_title(f'Densidad de {column}')
        plt.tight_layout()
        plt.show()

    # Codificación de variables categóricas
    label_encoder = LabelEncoder()
    dataset['id_eq'] = label_encoder.fit_transform(dataset['id_eq'])

    # Guardar dataset procesado
    dataset.to_csv('../data/processed/features_for_model.csv', index=False)
    print("Ingeniería de características completada. Archivo guardado en '../data/processed/features_for_model.csv'.")
