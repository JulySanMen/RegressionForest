# app.py
from flask import Flask, render_template, send_file, jsonify
import matplotlib.pyplot as plt
import io
from model import load_or_train_model
from utils import load_data, scale_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from io import StringIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
file_path = 'TotalFeatures-ISCXFlowMeter.csv'  # Especifica la ruta del archivo CSV
global df, X_train, y_train 
df, X_train, y_train = load_data(file_path)

@app.route('/evaluate')
def evaluate():
    X_train_scaled = scale_data(X_train)
    clf_rnd_scaled = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_rnd_scaled.fit(X_train_scaled, y_train)
    
    clf_rnd = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_rnd.fit(X_train, y_train)
    
    y_train_pred = clf_rnd.predict(X_train)
    y_train_prep_pred = clf_rnd_scaled.predict(X_train_scaled)
    
    f1_without_prep = f1_score(y_train, y_train_pred, average='weighted')
    f1_with_prep = f1_score(y_train, y_train_prep_pred, average='weighted')
    
    return jsonify({
        "f1_score_without_preparation": f1_without_prep,
        "f1_score_with_preparation": f1_with_prep
    })

@app.route('/showdatasets')
def show():
    # Convertir head, describe, y info a HTML y devolver como JSON
    data_head_html = df.head(10).to_html(classes="table table-striped")
    data_describe_html = df.describe().to_html(classes="table table-bordered")
    
    buffer = StringIO()
    df.info(buf=buffer)
    data_info_html = buffer.getvalue().replace('\n', '<br>')
    
    return jsonify({
        "data_head": data_head_html,
        "data_describe": data_describe_html,
        "data_info": data_info_html
    })

@app.route('/correlaciones')
def correlations():
    # Transformar la variable de salida y calcular la matriz de correlación
    X = df.copy()
    X['calss'] = X['calss'].factorize()[0]
    corr_matrix = X.corr()
    
    # Obtener las correlaciones con calss, ordenadas
    corr_class = corr_matrix["calss"].sort_values(ascending=False).to_frame()
    
    corr_html = corr_class.to_html(classes="table table-striped")
    
    return jsonify({"correlations": corr_html})

# Ruta para cargar y graficar datos
@app.route('/graficacion', methods=['GET'])
def index():
    # Cargar datos y entrenar/cargar el modelo
   
    X_train_scaled = scale_data(X_train)

    # Cargar o entrenar el modelo
    regressor, regressor_scaled, error_scaled, error_unscaled, y_train_encoded = load_or_train_model(X_train, y_train)
    
    # Obtener predicciones
    y_pred_unscaled = regressor.predict(X_train)
    y_pred_scaled = regressor_scaled.predict(X_train_scaled)

    # Generar gráficos
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Gráfico sin escalar
    ax[0].scatter(y_train_encoded, y_pred_unscaled, color='blue', label='Predicción sin Escalado')
    ax[0].plot([min(y_train_encoded), max(y_train_encoded)], [min(y_train_encoded), max(y_train_encoded)], color='red', linestyle='--')
    ax[0].set_xlabel('Valores Reales')
    ax[0].set_ylabel('Predicciones')
    ax[0].set_title('Predicción sin Escalado')
    ax[0].legend()

    # Gráfico con escalado
    ax[1].scatter(y_train_encoded, y_pred_scaled, color='green', label='Predicción con Escalado')
    ax[1].plot([min(y_train_encoded), max(y_train_encoded)], [min(y_train_encoded), max(y_train_encoded)], color='red', linestyle='--')
    ax[1].set_xlabel('Valores Reales')
    ax[1].set_ylabel('Predicciones')
    ax[1].set_title('Predicción con Escalado')
    ax[1].legend()

    plt.tight_layout()
    
    # Guardar el gráfico en un objeto en memoria para enviarlo como respuesta
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
