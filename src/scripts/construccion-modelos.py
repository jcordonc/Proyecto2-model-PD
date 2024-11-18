import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

def construccion-modelos():
    # Cargar el dataset (asegúrate de haber cargado tu archivo previamente)
    dataset = pd.read_csv('../data/processed/features_for_model.csv')

    # Definir X (características de entrada) y Y (variables de salida)
    X = dataset.drop(columns=['C3S', 'C2S', 'C3A', 'C4AF', 'FCAO', 'Unnamed: 0'])
    Y = dataset[['C3S', 'C2S', 'C3A', 'C4AF', 'FCAO']]

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Configuraciones de hiperparámetros para cada modelo
    models = {
        'LinearRegression': [
            {'fit_intercept': True},
            {'fit_intercept': False}
        ],
        'RandomForestRegressor': [
            {'n_estimators': 50, 'max_depth': 10},
            {'n_estimators': 100, 'max_depth': 20},
            {'n_estimators': 200, 'max_depth': None}
        ],
        'GradientBoostingRegressor': [
            {'n_estimators': 50, 'learning_rate': 0.1},
            {'n_estimators': 100, 'learning_rate': 0.05},
            {'n_estimators': 200, 'learning_rate': 0.01}
        ],
        'SVR': [
            {'kernel': 'linear', 'C': 1.0},
            {'kernel': 'rbf', 'C': 10.0},
            {'kernel': 'poly', 'degree': 2, 'C': 1.0}
        ],
        'KNeighborsRegressor': [
            {'n_neighbors': 5, 'weights': 'uniform'},
            {'n_neighbors': 10, 'weights': 'distance'},
            {'n_neighbors': 15, 'weights': 'uniform'}
        ]
    }

    # Entrenar y evaluar cada modelo con sus configuraciones
    results = []
    for model_name, param_list in models.items():
        for i, params in enumerate(param_list, 1):
            if model_name == 'LinearRegression':
                model = MultiOutputRegressor(LinearRegression(**params))
            elif model_name == 'RandomForestRegressor':
                model = MultiOutputRegressor(RandomForestRegressor(**params, random_state=42))
            elif model_name == 'GradientBoostingRegressor':
                model = MultiOutputRegressor(GradientBoostingRegressor(**params, random_state=42))
            elif model_name == 'SVR':
                model = MultiOutputRegressor(SVR(**params))
            elif model_name == 'KNeighborsRegressor':
                model = MultiOutputRegressor(KNeighborsRegressor(**params))
        
            # Entrenar el modelo
            model.fit(X_train, Y_train)
        
            # Realizar predicciones
            Y_pred = model.predict(X_test)
        
            # Calcular el MSE para cada salida
            mse_scores = mean_squared_error(Y_test, Y_pred, multioutput='raw_values')
        
            # Almacenar resultados
            results.append({
                'Model': model_name,
                'Configuration': f'Config {i}',
                'MSE_C3S': mse_scores[0],
                'MSE_C2S': mse_scores[1],
                'MSE_C3A': mse_scores[2],
                'MSE_C4AF': mse_scores[3],
                'MSE_FCAO': mse_scores[4]
            })

    # Mostrar los resultados en un DataFrame
    results_df = pd.DataFrame(results)
