from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
import os
import numpy as np
import joblib
import tensorflow as tf

# Map (coal_source, analysis_env) to model index
MODEL_MAP = {
    ("Central India Coal", "On Air Dried Basis"): 1,
    ("Central India Coal", "Moist on 60% RH & 40°C"): 2,
    ("Southern India Coal", "On Air Dried Basis"): 3,
    ("Southern India Coal", "Moist on 60% RH & 40°C"): 4,
}

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Cache loaded models to avoid reloading every request
LOADED_MODELS = {}

def load_models(i):
    if i in LOADED_MODELS:
        return LOADED_MODELS[i]
    # Load Keras embedder
    embedder = tf.keras.models.load_model(os.path.join(MODEL_DIR, f"gfen_embedder_model{i}.keras"))
    # Load scaler
    scaler = joblib.load(os.path.join(MODEL_DIR, f"scaler{i}.pkl"))
    # Load XGBoost model
    xgb = joblib.load(os.path.join(MODEL_DIR, f"xgb_model{i}.pkl"))
    LOADED_MODELS[i] = (embedder, scaler, xgb)
    return embedder, scaler, xgb

def home(request):
    return render(request, 'myapp/index.html')

@csrf_exempt
def predict_gcv(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            print("Received data:", data)
            coal_source = data.get('coalSource')
            analysis_env = data.get('analysisEnvironment')
            moisture = float(data.get('moisture', 0))
            ash = float(data.get('ash', 0))
            volatile_matter = float(data.get('volatileMatter', 0))
            fixed_carbon = float(data.get('fixedCarbon', 0))

            model_idx = MODEL_MAP.get((coal_source, analysis_env))
            if not model_idx:
                return JsonResponse({'error': 'Invalid combination of source and environment'}, status=400)

            embedder, scaler, xgb = load_models(model_idx)

            X = np.array([[moisture, ash, volatile_matter, fixed_carbon]], dtype=np.float32)
            X_scaled = scaler.transform(X)
            X_embed = embedder.predict(X_scaled)
            X_combined = np.concatenate([X_scaled, X_embed], axis=1)  # <-- FIXED LINE
            print("X_scaled shape:", X_scaled.shape)
            print("X_embed shape:", X_embed.shape)
            print("X_combined shape:", X_combined.shape)
            print("XGBoost expects:", xgb.n_features_in_)

            gcv_pred = xgb.predict(X_combined)[0]

            print("Predicted GCV:", gcv_pred)

            return JsonResponse({
                'gcv': round(float(gcv_pred), 2),
                'inputs': {
                    'coalSource': coal_source,
                    'analysisEnvironment': analysis_env,
                    'moisture': moisture,
                    'ash': ash,
                    'volatileMatter': volatile_matter,
                    'fixedCarbon': fixed_carbon
                }
            })
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return JsonResponse({'error': f'Internal error: {str(e)}'}, status=500)
    return JsonResponse({'error': 'Invalid request'}, status=400)