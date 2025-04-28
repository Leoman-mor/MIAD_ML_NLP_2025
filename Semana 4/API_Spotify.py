# spotify_api_local.py - API para predicción de popularidad usando archivos locales

import pandas as pd
import numpy as np
import joblib
import json
import os
from flask import Flask
from flask_restx import Api, Resource, fields, reqparse

# Inicializar la aplicación Flask y la API
app = Flask(__name__)
api = Api(app, 
          version='1.0', 
          title='API de Predicción de Popularidad de Canciones',
          description='API para predecir la popularidad de canciones de Spotify')

# Definir el espacio de nombres para los endpoints
ns = api.namespace('spotify', description='Predicción de popularidad')

# Rutas a los archivos del modelo (en la misma carpeta que este script)
MODEL_PATH = 'spotify_popularity_model.pkl'
FEATURE_LIST_PATH = 'feature_list.json'

# Verificar que los archivos existen
if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_LIST_PATH):
    print(f"ERROR: No se encontraron los archivos del modelo:")
    print(f"  - Modelo: {'✓' if os.path.exists(MODEL_PATH) else '✗'} ({MODEL_PATH})")
    print(f"  - Características: {'✓' if os.path.exists(FEATURE_LIST_PATH) else '✗'} ({FEATURE_LIST_PATH})")
    print("Asegúrate de que los archivos estén en la misma carpeta que este script.")
    model = None
    feature_list = None
else:
    # Cargar el modelo entrenado
    print(f"Cargando modelo desde {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    
    # Cargar la lista de características
    print(f"Cargando lista de características desde {FEATURE_LIST_PATH}...")
    with open(FEATURE_LIST_PATH, 'r') as f:
        feature_list = json.load(f)
    
    print(f"Modelo cargado con {len(feature_list)} características")

# Definir el modelo de datos para la respuesta
resource_fields = api.model('PredictionResult', {
    'result': fields.Float(description='Popularidad predicha (0-100)'),
})

# Crear el parser para los argumentos de entrada
parser = reqparse.RequestParser()

# Añadir todos los argumentos necesarios basados en las características del modelo
if feature_list:
    for feature in feature_list:
        parser.add_argument(feature, type=float, required=True, 
                          help=f'Valor de {feature} requerido')

# Función para predicción
def predict_popularity(song_features):
    """Predice la popularidad de una canción basada en sus características"""
    # Crear DataFrame con los valores de entrada
    input_df = pd.DataFrame([song_features])
    
    # Asegurar que las columnas estén en el orden correcto
    input_df = input_df[feature_list]
    
    # Realizar la predicción
    prediction = model.predict(input_df)[0]
    
    # Asegurar que la predicción esté en el rango 0-100
    prediction = float(np.clip(prediction, 0, 100))
    
    return round(prediction, 2)

# Definición de la clase para disponibilización
@ns.route('/')
class SpotifyPopularityApi(Resource):
    
    @ns.doc(parser=parser)
    @ns.marshal_with(resource_fields)
    def get(self):
        """Predice la popularidad de una canción basada en sus características"""
        # Verificar si el modelo está cargado
        if model is None or feature_list is None:
            api.abort(500, "El modelo no está disponible. Verifica los logs del servidor.")
            
        # Obtener todos los argumentos
        args = parser.parse_args()
        
        # Extraer los valores para la predicción
        song_features = {}
        for feature in feature_list:
            song_features[feature] = args[feature]
            
        # Realizar la predicción
        popularity = predict_popularity(song_features)
        
        # Devolver el resultado
        return {
            "result": popularity
        }, 200

if __name__ == '__main__':
    # Ejecución de la aplicación que disponibiliza el modelo
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
