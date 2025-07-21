import joblib
import numpy as np
from preprocessing import preprocess_input

def load_models():
    """Load base and meta models from disk."""
    models = {
        'xgb': joblib.load("models/final_xgboost.pkl"),
        'lgbm': joblib.load("models/final_lightgbm.pkl"),
        'meta': joblib.load("models/meta_model_logreg.pkl")
    }
    return models

def predict_fraud(input_data):
    """Given raw input, return fraud prediction using ensemble."""
    preprocessed = preprocess_input(input_data)
    models = load_models()

    # Base model predictions (probabilities)
    xgb_pred = models['xgb'].predict_proba(preprocessed)[:, 1]
    lgbm_pred = models['lgbm'].predict_proba(preprocessed)[:, 1]


    # Stack into meta input
    meta_input = np.column_stack([xgb_pred, lgbm_pred])

    # Final prediction
    final_pred = models['meta'].predict(meta_input)
    final_proba = models['meta'].predict_proba(meta_input)[:, 1]

    return {
        'prediction': int(final_pred[0]),
        'probability': float(final_proba[0]),
        'model_probs': {
            'xgb': float(xgb_pred[0]),
            'lgbm': float(lgbm_pred[0])                                                                                                                                 
        }
    }

def get_shap_input(input_data):
    """Returns the LightGBM model and preprocessed input for SHAP analysis."""
    preprocessed = preprocess_input(input_data)
    lgbm_model = joblib.load("models/final_lightgbm.pkl")
    return lgbm_model, preprocessed
