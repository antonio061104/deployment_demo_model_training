from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(model, test_data, y_test):
    """
    Evalúa el modelo en los datos de prueba.

    Args:
        model: Modelo entrenado.
        test_data (pd.DataFrame): Datos de prueba.
        y_test (pd.Series): Etiquetas de prueba.

    Returns:
        tuple: Accuracy, Precision, Recall, F1, AUC.
    """
    y_pred = model.predict(test_data)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    auc = roc_auc_score(y_test, model.predict_proba(test_data), multi_class='ovr')

    return accuracy, precision, recall, f1, auc


    