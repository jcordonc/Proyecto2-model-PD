"""
Este módulo crea y procesa un proceso de ingeniería de características para entrenar y probar conjuntos de datos.
Guarda los conjuntos de datos procesados y el pipeline para su uso posterior.
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from feature_engine.encoding import OrdinalEncoder


def create_features_pipeline():
    """
    Crea y procesa un proceso de ingeniería de características para entrenar y probar conjuntos de datos.
    Guarda los conjuntos de datos procesado y pipeline para su uso posterior.
    """
    # load dataset
    project_path = os.getcwd()
    dataset = pd.read_csv(os.path.join(
        project_path, 'data', 'raw', 'dataset_train.csv'))

    # define x (input features) and y (output variables)
    x = dataset.drop(columns=['C3S', 'C2S', 'C3A',
                     'C4AF', 'FCAO', 'Unnamed: 0'])
    y = dataset[['C3S', 'C2S', 'C3A', 'C4AF', 'FCAO']]

    # split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, shuffle=True
    )

    # create the pipeline
    pipeline = Pipeline([
        ('ordinal_encoder', OrdinalEncoder(
            encoding_method='arbitrary', variables=['id_eq'])),
        ('scaler', StandardScaler())
    ])

    # fit the pipeline on training data
    pipeline.fit(x_train)

    # transform training features
    x_features_train = pipeline.transform(x_train)
    df_features_train = pd.DataFrame(x_features_train, columns=x_train.columns)

    # reset indices for consistency
    df_features_train = df_features_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    # combine transformed features and target variables
    df_features_train = pd.concat([df_features_train, y_train], axis=1)

    # reset indices for testing data
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # combine testing features and target variables
    dataset_test = pd.concat([x_test, y_test], axis=1)

    # save processed datasets
    df_features_train.to_csv(os.path.join(
        project_path, "data", "processed", "df_features_train.csv"), index=False)
    dataset_test.to_csv(os.path.join(project_path, "data",
                        "processed", "dataset_test.csv"), index=False)

    # save the pipeline
    with open(os.path.join(project_path, 'artifacts', 'pipeline.pkl'), 'wb') as f:
        pickle.dump(pipeline, f)


create_features_pipeline()
