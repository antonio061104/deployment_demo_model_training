from datetime import datetime
import joblib
import xgboost as xgb

def save_model(model: xgb.XGBClassifier, model_path: str) -> None:
    """
    Guarda el modelo en un archivo usando joblib con un timestamp.

    Args:
        model (xgb.XGBClassifier): Modelo a guardar.
        model_path (str): Ruta base donde guardar el modelo (sin extensión).
    """
    current_date = datetime.now().strftime("%Y-%m-%d")
    full_path = f"{model_path}_{current_date}.joblib"
    joblib.dump(model, full_path)
    print(f"Modelo guardado en {full_path}")
