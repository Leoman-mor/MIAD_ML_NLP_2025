# spotify_api_github.py - API para predicción de popularidad usando modelo desde GitHub

import pandas as pd
import numpy as np
import joblib
import json
import requests
from io import BytesIO
from flask import Flask
from flask_restx import Api, Resource, fields, reqparse
import os
import tempfile

# URLs de GitHub donde están almacenados los archivos
MODEL_URL = "https://github.com/Leoman-mor/MIAD_ML_NLP_2025/blob/main/Semana%204/spotify_popularity_model.pkl"
FEATURE_LIST_URL = "https://github.com/Leoman-mor/MIAD_ML_NLP_2025/blob/main/Semana%204/feature_list.json"

# Inicializar la aplicación Flask y la API
app = Flask(__name__)
api = Api(app, 
          version='1.0', 
          title='API de Predicción de Popularidad de Canciones',
          description='API para predecir la popularidad de canciones de Spotify')

# Definir el espacio de nombres para los endpoints
ns = api.namespace('spotify', description='Predicción de popularidad')

def download_from_github(url):
    """Descarga un archivo desde GitHub"""
    response = requests.get(url)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        raise Exception(f"Error descargando desde GitHub: {response.status_code}")

# Cargar el modelo y features desde GitHub
print("Descargando modelo desde GitHub...")
model = None
feature_list = None

try:
    # Crear un archivo temporal para el modelo
    with tempfile.NamedTemporaryFile(delete=False) as tmp_model_file:
        tmp_model_path = tmp_model_file.name
        # Descargar el contenido del modelo
        model_content = download_from_github(MODEL_URL)
        # Escribir el contenido al archivo temporal
        tmp_model_file.write(model_content.getbuffer())
    
    # Cargar el modelo desde el archivo temporal
    model = joblib.load(tmp_model_path)
    # Eliminar el archivo temporal
    os.unlink(tmp_model_path)
    
    # Descargar la lista de características
    feature_list_content = download_from_github(FEATURE_LIST_URL)
    feature_list = json.loads(feature_list_content.getvalue().decode('utf-8'))
    
    print(f"Modelo cargado con {len(feature_list)} características")
    
except Exception as e:
    print(f"Error cargando el modelo desde GitHub: {str(e)}")
    print("La API no podrá funcionar correctamente sin el modelo")

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