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

# Ash range to average volatile matter mapping for model 1
ASH_VM_TABLE_1 = [
    (12.9, 17.89, 29.46),
    (17.9, 22.89, 28.44),
    (22.9, 27.89, 26.87),
    (27.9, 32.89, 25.62),
    (32.9, 37.89, 24.07),
    (37.9, 42.89, 22.96),
    (42.9, 47.89, 21.63),
    (47.9, 52.89, 20.67),
    (52.9, 57.89, 18.63),
    (57.9, 62.89, 16.22),
    (62.9, 67.89, 20.99),
    (67.9, 72.89, 12.17),
]

# Ash range to average volatile matter mapping for model 2
ASH_VM_TABLE_2 = [
    (13.05, 18.04, 29.81),
    (18.05, 23.04, 28.62),
    (23.05, 28.04, 26.83),
    (28.05, 33.04, 25.54),
    (33.05, 38.04, 23.85),
    (38.05, 43.04, 22.83),
    (43.05, 48.04, 21.43),
    (48.05, 53.04, 19.94),
    (53.05, 58.04, 18.70),
    (58.05, 63.04, 16.02),
    (63.05, 68.04, 14.91),
    (68.05, 73.04, 12.16),
]

# Ash range to average volatile matter mapping for model 3
ASH_VM_TABLE_3 = [
    (11.88, 16.87, 29.58),
    (16.88, 21.87, 29.52),
    (21.88, 26.87, 27.65),
    (26.88, 31.87, 25.61),
    (31.88, 36.87, 25.35),
    (36.88, 41.87, 23.55),
    (41.88, 46.87, 22.49),
    (46.88, 51.87, 20.18),
    (51.88, 56.87, 18.49),
    (56.88, 61.87, 17.56),
    (61.88, 66.87, 16.98),
    (11.88, 16.87, 16.32),  # Note: This range is repeated in your table; you may want to clarify
]

# Ash range to average volatile matter mapping for model 4
ASH_VM_TABLE_4 = [
    (11.98, 16.97, 29.76),
    (16.98, 21.97, 29.69),
    (21.98, 26.97, 27.85),
    (26.98, 31.97, 25.81),
    (31.98, 36.97, 25.13),
    (36.98, 41.97, 23.60),
    (41.98, 46.97, 22.65),
    (46.98, 51.97, 19.91),
    (51.98, 56.97, 18.40),
    (56.98, 61.97, 17.63),
    (61.98, 66.97, 16.68),
    (66.98, 71.97, 14.79),
    (71.98, 76.97, 14.79),
    (76.98, 81.97, 14.79),
    (81.98, 86.97, 11.94),
]

def get_default_vm(ash, model_idx):
    if model_idx == 1:
        table = ASH_VM_TABLE_1
    elif model_idx == 2:
        table = ASH_VM_TABLE_2
    elif model_idx == 3:
        table = ASH_VM_TABLE_3
    elif model_idx == 4:
        table = ASH_VM_TABLE_4
    else:
        return 20.0  # fallback default
    for low, high, avg_vm in table:
        if low <= ash <= high:
            return avg_vm
    return 20.0  # fallback default

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
            volatile_matter = data.get('volatileMatter')
            fixed_carbon = data.get('fixedCarbon')
            fixed_carbon_defaulted = False
            volatile_matter_defaulted = False

            model_idx = MODEL_MAP.get((coal_source, analysis_env))
            if not model_idx:
                return JsonResponse({'error': 'Invalid combination of source and environment'}, status=400)

            # Handle volatile matter defaulting
            if volatile_matter is None or volatile_matter == "":
                volatile_matter = get_default_vm(ash, model_idx)
                volatile_matter_defaulted = True
            else:
                volatile_matter = float(volatile_matter)

            # Handle fixed carbon defaulting/recalculation
            if fixed_carbon is None or fixed_carbon == "":
                fixed_carbon = 100 - moisture - ash - volatile_matter
                fixed_carbon_defaulted = True
            else:
                fixed_carbon = float(fixed_carbon)

            embedder, scaler, xgb = load_models(model_idx)

            X = np.array([[moisture, ash, volatile_matter, fixed_carbon]], dtype=np.float32)
            X_scaled = scaler.transform(X)
            X_embed = embedder.predict(X_scaled)
            X_combined = np.concatenate([X_scaled, X_embed], axis=1)

            gcv_pred = xgb.predict(X_combined)[0]

            return JsonResponse({
                'gcv': round(float(gcv_pred), 2),
                'inputs': {
                    'coalSource': coal_source,
                    'analysisEnvironment': analysis_env,
                    'moisture': moisture,
                    'ash': ash,
                    'volatileMatter': volatile_matter,
                    'fixedCarbon': fixed_carbon
                },
                'volatileMatterDefaulted': volatile_matter_defaulted,
                'fixedCarbonDefaulted': fixed_carbon_defaulted
            })
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return JsonResponse({'error': f'Internal error: {str(e)}'}, status=500)
    return JsonResponse({'error': 'Invalid request'}, status=400)