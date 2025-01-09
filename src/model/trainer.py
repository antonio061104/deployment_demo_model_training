import xgboost as xgb

def train_model(X_train, y_train):
    """
    Entrena un modelo XGBoost en los datos de entrenamiento.

    Args:
        X_train (pd.DataFrame): Datos de entrenamiento.
        y_train (pd.Series): Etiquetas de entrenamiento.

    Returns:
        xgb.XGBClassifier: Modelo entrenado.
    """
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    return model
