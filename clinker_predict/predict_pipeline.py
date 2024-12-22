"""
Este módulo carga una canalización entrenada, hace predicciones sobre nuevos datos de prueba,
guarda las predicciones como CSV y registra los resultados utilizando MLflow.
"""

import os
import datetime
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn


def predict_pipeline():
    """
    Carga un pipeline, hace predicciones sobre nuevos datos de prueba, guarda las predicciones,
    y registra los resultados utilizando MLflow.
    """
    # Load trained pipeline
    project_path = os.getcwd()
    with open(os.path.join(project_path, 'artifacts', 'pipeline_trained.pkl'), 'rb') as f:
        pipeline = pickle.load(f)

    # Load raw test data
    test_new_data = pd.read_csv(os.path.join(
        project_path, 'data', 'raw', 'dataset_test.csv'))
    test_new_data.drop(['Unnamed: 0'], axis=1, inplace=True)

    # Make predictions
    predicts_array = pipeline.predict(test_new_data)
    predicts = pd.DataFrame(predicts_array, columns=[
                            'C3S', 'C2S', 'C3A', 'C4AF', 'FCAO'])

    # Generate filename with current timestamp
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"{current_time}.csv"

    # Save predictions to CSV
    file_path = os.path.join(project_path, 'data', 'predictions', filename)
    predicts.to_csv(file_path, index=False)

    # Configure MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("Predicts_Cement")

    # Log artifact with MLflow
    with mlflow.start_run():
        mlflow.log_artifact(file_path, artifact_path="predictions")
    mlflow.end_run()


predict_pipeline()
