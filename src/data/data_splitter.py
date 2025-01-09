from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(data: pd.DataFrame, target_column: str, test_size=0.25, 
               random_state=42, stratify: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide los datos del dataset de Cleveland Heart Disease en conjuntos de entrenamiento y prueba.

    Args:
        data (pd.DataFrame): DataFrame que contiene los datos a dividir.
        target_column (str): Nombre de la columna objetivo ('target').
        test_size (float, optional): Proporción de los datos en el conjunto de prueba. Defaults to 0.25.
        random_state (int, optional): Semilla para la aleatoriedad. Defaults to 42.
        stratify (bool, optional): Si estratificar los datos según la variable objetivo. Defaults to True.

    Returns:
        tuple: Conjuntos de datos divididos:
            - X_train (pd.DataFrame): Variables independientes para entrenamiento.
            - X_test (pd.DataFrame): Variables independientes para prueba.
            - y_train (pd.Series): Variable objetivo para entrenamiento.
            - y_test (pd.Series): Variable objetivo para prueba.
    """
    X = data.drop(columns=target_column, axis=1)
    y = data[target_column]

    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    return X_train, X_test, y_train, y_test
