"""
Este script entrena y evalúa modelos de aprendizaje automático para la predicción de la calidad del cemento.
Lee conjuntos de datos de prueba y entrenamiento preprocesados, configura hiperparámetros,
selecciona el mejor modelo basándose en el error cuadrático medio (MSE) e integra el modelo
a una tubería preexistente. Finalmente, el pipeline se guarda para predicciones posteriores.
"""

import os
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


def configure_and_train_pipeline():
    """
    Configura, entrena y evalúa múltiples modelos de aprendizaje automático,
    selecciona el mejor modelo y lo integra a una tubería existente.
    Guarda la tubería para su uso posterior.
    """
    project_path = os.getcwd()
    df_features_train = pd.read_csv(os.path.join(
        project_path, "data", "processed", "df_features_train.csv"))
    dataset_test = pd.read_csv(os.path.join(project_path, "data",
                                            "processed", "dataset_test.csv"))

    # Define X (input features) and Y (output variables)
    x_features_train = df_features_train.drop(
        columns=["C3S", "C2S", "C3A", "C4AF", "FCAO"]
    )
    y_train = df_features_train[["C3S", "C2S", "C3A", "C4AF", "FCAO"]]

    x_test = dataset_test.drop(columns=["C3S", "C2S", "C3A", "C4AF", "FCAO"])
    y_test = dataset_test[["C3S", "C2S", "C3A", "C4AF", "FCAO"]]

    # Load pre-configured pipeline
    with open(os.path.join(project_path, 'artifacts', 'pipeline.pkl'), "rb") as f:
        pipeline = pickle.load(f)

    # Transform test data using the pipeline
    x_features_test_array = pipeline.transform(x_test)
    x_features_test = pd.DataFrame(
        x_features_test_array, columns=x_test.columns
    )

    # Configure hyperparameters for models
    models = {
        "LinearRegression": [
            {"fit_intercept": True},
            {"fit_intercept": False}
        ],
        "RandomForestRegressor": [
            {"n_estimators": 50, "max_depth": 10},
            {"n_estimators": 100, "max_depth": 20},
            {"n_estimators": 200, "max_depth": None}
        ],
        "GradientBoostingRegressor": [
            {"n_estimators": 50, "learning_rate": 0.1},
            {"n_estimators": 100, "learning_rate": 0.05},
            {"n_estimators": 200, "learning_rate": 0.01}
        ],
        "SVR": [
            {"kernel": "linear", "C": 1.0},
            {"kernel": "rbf", "C": 10.0},
            {"kernel": "poly", "degree": 2, "C": 1.0}
        ],
        "KNeighborsRegressor": [
            {"n_neighbors": 5, "weights": "uniform"},
            {"n_neighbors": 10, "weights": "distance"},
            {"n_neighbors": 15, "weights": "uniform"}
        ]
    }

    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("Model_Selection_Cement")

    # Train and evaluate models
    results = []

    for model_name, param_list in models.items():
        for i, params in enumerate(param_list, 1):
            with mlflow.start_run(run_name=f"{model_name}_Config_{i}"):
                if model_name == "LinearRegression":
                    model = MultiOutputRegressor(LinearRegression(**params))
                elif model_name == "RandomForestRegressor":
                    model = MultiOutputRegressor(
                        RandomForestRegressor(**params, random_state=42)
                    )
                elif model_name == "GradientBoostingRegressor":
                    model = MultiOutputRegressor(
                        GradientBoostingRegressor(**params, random_state=42)
                    )
                elif model_name == "SVR":
                    model = MultiOutputRegressor(SVR(**params))
                elif model_name == "KNeighborsRegressor":
                    model = MultiOutputRegressor(KNeighborsRegressor(**params))

                model.fit(x_features_train, y_train)

                y_pred = model.predict(x_features_test)
                mse_scores = mean_squared_error(
                    y_test, y_pred, multioutput="raw_values"
                )

                avg_mse = np.mean(mse_scores)

                mlflow.log_params(params)
                mlflow.log_metric("MSE_C3S", mse_scores[0])
                mlflow.log_metric("MSE_C2S", mse_scores[1])
                mlflow.log_metric("MSE_C3A", mse_scores[2])
                mlflow.log_metric("MSE_C4AF", mse_scores[3])
                mlflow.log_metric("MSE_FCAO", mse_scores[4])
                mlflow.log_metric("MSE_Average", avg_mse)

                mlflow.sklearn.log_model(model, artifact_path="model")

                results.append({
                    "Model": model_name,
                    "Configuration": f"Config {i}",
                    "MSE_C3S": mse_scores[0],
                    "MSE_C2S": mse_scores[1],
                    "MSE_C3A": mse_scores[2],
                    "MSE_C4AF": mse_scores[3],
                    "MSE_FCAO": mse_scores[4],
                    "MSE_Average": avg_mse
                })
    mlflow.end_run()

    results_df = pd.DataFrame(results)

    # Calculate average MSE and find the best model
    results_df["MSE_Average"] = results_df[[
        "MSE_C3S", "MSE_C2S", "MSE_C3A", "MSE_C4AF", "MSE_FCAO"
    ]].mean(axis=1)

    best_result = results_df.loc[results_df["MSE_Average"].idxmin()]

    best_model_name = best_result["Model"]
    best_config_index = int(best_result["Configuration"].split(" ")[-1]) - 1
    best_config = models[best_model_name][best_config_index]

    print(f"Best Model: {best_model_name}")
    print(f"Best Configuration: {best_config}")
    print(f"Average MSE: {best_result['MSE_Average']}")

    if best_model_name == "LinearRegression":
        best_model = MultiOutputRegressor(LinearRegression(**best_config))
    elif best_model_name == "RandomForestRegressor":
        best_model = MultiOutputRegressor(
            RandomForestRegressor(**best_config, random_state=42)
        )
    elif best_model_name == "GradientBoostingRegressor":
        best_model = MultiOutputRegressor(
            GradientBoostingRegressor(**best_config, random_state=42)
        )
    elif best_model_name == "SVR":
        best_model = MultiOutputRegressor(SVR(**best_config))
    elif best_model_name == "KNeighborsRegressor":
        best_model = MultiOutputRegressor(KNeighborsRegressor(**best_config))

    pipeline.steps.append(("regression_model", best_model))

    # Load full training data and retrain the model
    dataset_train = pd.read_csv(os.path.join(
        project_path, 'data', 'raw', 'dataset_train.csv'))
    dataset_train.drop(["Unnamed: 0"], axis=1, inplace=True)
    dataset_train_features = dataset_train.drop(
        columns=["C3S", "C2S", "C3A", "C4AF", "FCAO"], axis=1
    )
    dataset_train_target = dataset_train[["C3S", "C2S", "C3A", "C4AF", "FCAO"]]

    pipeline.fit(dataset_train_features, dataset_train_target)

    with open(os.path.join(project_path, 'artifacts', 'pipeline_trained.pkl'), "wb") as f:
        pickle.dump(pipeline, f)


configure_and_train_pipeline()
