import os
import pandas as pd

def load_data(file_path="C:/Users/anton/OneDrive/Documentos/GitHub/deployment_demo_model_training/data/raw/processed.cleveland.data"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo no existe: {file_path}")
    print(f"Cargando archivo desde: {file_path}")
    df = pd.read_csv(file_path, header=None)
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    df.columns = column_names
    return df
