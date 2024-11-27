"""
Este script entrena varios modelos de regresión para predecir múltiples salidas relacionadas
con la composición química del clinker. También selecciona y guarda el mejor modelo basado en MSE.
"""

import pickle
import pandas as pd
import ace_tools
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def construir_modelos():
    """
    Entrena varios modelos de regresión para predecir variables químicas (C3S, C2S, C3A, C4AF, 
    FCAO).
    Selecciona el mejor modelo basado en el MSE promedio y lo guarda junto con el StandardScaler.
    """
    # Cargar el dataset
    dataset = pd.read_csv('../data/processed/features_for_model.csv')

    # Definir características de entrada (X) y variables de salida (y)
    features = dataset.drop(columns=['C3S', 'C2S', 'C3A', 'C4AF', 'FCAO', 'Unnamed: 0'])
    targets = dataset[['C3S', 'C2S', 'C3A', 'C4AF', 'FCAO']]

    # Dividir los datos en conjuntos de entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )

    # Escalar los datos
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Guardar el escalador
    with open('../artifacts/std_scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

    # Configuración de modelos e hiperparámetros
    modelos_config = {
        'LinearRegression': [{'fit_intercept': True}, {'fit_intercept': False}],
        'RandomForestRegressor': [
            {'n_estimators': 50, 'max_depth': 10},
            {'n_estimators': 100, 'max_depth': 20},
            {'n_estimators': 200, 'max_depth': None},
        ],
        'GradientBoostingRegressor': [
            {'n_estimators': 50, 'learning_rate': 0.1},
            {'n_estimators': 100, 'learning_rate': 0.05},
            {'n_estimators': 200, 'learning_rate': 0.01},
        ],
        'SVR': [
            {'kernel': 'linear', 'C': 1.0},
            {'kernel': 'rbf', 'C': 10.0},
            {'kernel': 'poly', 'degree': 2, 'C': 1.0},
        ],
        'KNeighborsRegressor': [
            {'n_neighbors': 5, 'weights': 'uniform'},
            {'n_neighbors': 10, 'weights': 'distance'},
            {'n_neighbors': 15, 'weights': 'uniform'},
        ],
    }

    # Entrenar y evaluar modelos
    resultados = []
    for nombre_modelo, configuraciones in modelos_config.items():
        for idx, params in enumerate(configuraciones, 1):
            if nombre_modelo == 'LinearRegression':
                modelo = MultiOutputRegressor(LinearRegression(**params))
            elif nombre_modelo == 'RandomForestRegressor':
                modelo = MultiOutputRegressor(RandomForestRegressor(**params, random_state=42))
            elif nombre_modelo == 'GradientBoostingRegressor':
                modelo = MultiOutputRegressor(GradientBoostingRegressor(**params, random_state=42))
            elif nombre_modelo == 'SVR':
                modelo = MultiOutputRegressor(SVR(**params))
            elif nombre_modelo == 'KNeighborsRegressor':
                modelo = MultiOutputRegressor(KNeighborsRegressor(**params))

            modelo.fit(x_train_scaled, y_train)
            predicciones = modelo.predict(x_test_scaled)
            mse_scores = mean_squared_error(y_test, predicciones, multioutput='raw_values')

            resultados.append({
                'Modelo': nombre_modelo,
                'Configuracion': f'Config {idx}',
                'MSE_C3S': mse_scores[0],
                'MSE_C2S': mse_scores[1],
                'MSE_C3A': mse_scores[2],
                'MSE_C4AF': mse_scores[3],
                'MSE_FCAO': mse_scores[4],
            })

    # Crear un DataFrame de resultados
    resultados_df = pd.DataFrame(resultados)
    resultados_df['MSE_promedio'] = resultados_df[
        ['MSE_C3S', 'MSE_C2S', 'MSE_C3A', 'MSE_C4AF', 'MSE_FCAO']
    ].mean(axis=1)

    # Seleccionar el mejor modelo
    mejor_fila = resultados_df.loc[resultados_df['MSE_promedio'].idxmin()]
    mejor_modelo = mejor_fila['Modelo']
    mejor_config_idx = int(mejor_fila['Configuracion'].split()[-1]) - 1

    # Mostrar resultados
    ace_tools.display_dataframe_to_user(name="Resultados de Modelos", dataframe=resultados_df)
    print(f"Mejor modelo: {mejor_modelo}, Configuración: Config {mejor_config_idx + 1}")

    # Guardar el mejor modelo
    mejor_params = modelos_config[mejor_modelo][mejor_config_idx]
    if mejor_modelo == 'LinearRegression':
        modelo_final = MultiOutputRegressor(LinearRegression(**mejor_params))
    elif mejor_modelo == 'RandomForestRegressor':
        modelo_final = MultiOutputRegressor(RandomForestRegressor(**mejor_params, random_state=42))
    elif mejor_modelo == 'GradientBoostingRegressor':
        modelo_final = MultiOutputRegressor(GradientBoostingRegressor(**mejor_params,
                                                                       random_state=42))
    elif mejor_modelo == 'SVR':
        modelo_final = MultiOutputRegressor(SVR(**mejor_params))
    elif mejor_modelo == 'KNeighborsRegressor':
        modelo_final = MultiOutputRegressor(KNeighborsRegressor(**mejor_params))

    modelo_final.fit(x_train_scaled, y_train)

    with open('../models/mejor_modelo.pkl', 'wb') as file:
        pickle.dump(modelo_final, file)

