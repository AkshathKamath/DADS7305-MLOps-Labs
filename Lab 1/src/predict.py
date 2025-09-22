import joblib

def predict_data(X):
    """
    """
    model = joblib.load("./model/wine_model.pkl")
    y_pred = model.predict(X)
    return y_pred