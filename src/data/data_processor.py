import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np


def process_data(df: pd.DataFrame, columns_to_impute: list, target_column: str):
    """
    Procesa los datos:
    - Imputa los valores faltantes
    - Escala las variables numéricas

    Args:
       df (pd.DataFrame): DataFrame con los datos a procesar.
       columns_to_impute (list): Columnas a procesar para imputar valores faltantes.
       target_column (str, optional): Columna objetivo, si aplica.

    Returns:
       tuple[pd.DataFrame, pd.Series]: DataFrame procesado y columna objetivo si se especifica.
    """

    # Convertir columnas 'ca' y 'thal' a numéricas
    df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
    df['thal'] = pd.to_numeric(df['thal'], errors='coerce')

    # Imputar valores faltantes
    df[columns_to_impute] = df[columns_to_impute].fillna(df[columns_to_impute].mean())

    # Escalar las variables numéricas
    scaler = StandardScaler()
    df[columns_to_impute] = scaler.fit_transform(df[columns_to_impute])

    # Separar la columna objetivo si se especifica
    target = df[target_column] if target_column else None

    return df, target

  

