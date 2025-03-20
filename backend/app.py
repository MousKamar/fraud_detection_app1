import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.keras
import tensorflow as tf
from flask_cors import CORS
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import dagshub
import dvc.api
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

TRAINING_FILES_FOLDER = os.path.join(os.getcwd(), 'training_files')
app.config["TRAINING_FILES_FOLDER"] = TRAINING_FILES_FOLDER


if not os.path.exists(TRAINING_FILES_FOLDER):
    os.makedirs(TRAINING_FILES_FOLDER)


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

dagshub.init(repo_owner='kamarmossa', repo_name='ByPassFraudNN', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/kamarmossa/ByPassFraudNN.mlflow")

def load_model_from_mlflow(model_name, model_version=None):
    try:
        if model_version:
            model_uri = f"models:/{model_name}/{model_version}"
        else:
            model_uri = f"models:/{model_name}/latest"

        model = mlflow.keras.load_model(model_uri)
        return model
    except Exception as e:
        return f"Error loading model: {str(e)}"

def Training_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.astype('str')
    df.fillna(0, inplace=True)

    df['sum_in_duration'] = df['DurationInSeconds'].where(df['RecordDirection'] == 2)
    df['count_in_distinctA'] = df['OtherPartyNumber'].where(df['RecordDirection'] == 1)

    df_data_to_analyze_grouped = df.groupby(['MSISDN']).agg(
        count_calls_all=('MSISDN', 'count'),
        count_distinct_A=('count_in_distinctA', 'nunique'),
        count_calls_in=('RecordDirection', lambda x: (x == 2).sum()),
        sum_duration_in=('sum_in_duration', lambda x: x.sum() / 60),
        sum_dura_all=('DurationInSeconds', lambda x: x.sum() / 60),
        count_calls_out=('RecordDirection', lambda x: (x == 1).sum()),
        fraud=('Fraud', 'max')
    ).reset_index()

    df_data_to_analyze_grouped['ratio_distinct_A'] = df_data_to_analyze_grouped['count_distinct_A'] / df_data_to_analyze_grouped['count_calls_out']
    df_data_to_analyze_grouped['ratio_in_all'] = df_data_to_analyze_grouped['count_calls_in'] / df_data_to_analyze_grouped['count_calls_all']
    df_data_to_analyze_grouped['ratio_in_duration'] = df_data_to_analyze_grouped['sum_duration_in'] / df_data_to_analyze_grouped['sum_dura_all']

    df_data_to_analyze_grouped.fillna(0, inplace=True)
    df_data_to_analyze_grouped = df_data_to_analyze_grouped.sort_values(by='count_calls_all', ascending=False)

    fraud = df_data_to_analyze_grouped.pop('fraud')
    df_data_to_analyze_grouped.drop(columns=['MSISDN'], inplace=True)

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_data_to_analyze_grouped)

    undersampler = RandomUnderSampler(sampling_strategy=0.6, random_state=42)
    X_train, Y_train = undersampler.fit_resample(df_scaled, fraud)

    xtrain, xvalid, ytrain, yvalid = train_test_split(X_train, Y_train, test_size=0.3, random_state=42)

    reshaped_images = xtrain.reshape(xtrain.shape[0], 3, 3, 1)  # 3x3 image with 1 channel

    return reshaped_images, xvalid, ytrain, yvalid

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.astype('str')
    df.fillna(0, inplace=True)

    df['sum_in_duration'] = df['DurationInSeconds'].where(df['RecordDirection'] == 2)
    df['count_in_distinctA'] = df['OtherPartyNumber'].where(df['RecordDirection'] == 1)

    df_data_to_analyze_grouped = df.groupby(['MSISDN']).agg(
        count_calls_all=('MSISDN', 'count'),
        count_distinct_A=('count_in_distinctA', 'nunique'),
        count_calls_in=('RecordDirection', lambda x: (x == 2).sum()),
        sum_duration_in=('sum_in_duration', lambda x: x.sum() / 60),
        sum_dura_all=('DurationInSeconds', lambda x: x.sum() / 60),
        count_calls_out=('RecordDirection', lambda x: (x == 1).sum())
    ).reset_index()

    df_data_to_analyze_grouped['ratio_distinct_A'] = df_data_to_analyze_grouped['count_distinct_A'] / df_data_to_analyze_grouped['count_calls_out']
    df_data_to_analyze_grouped['ratio_in_all'] = df_data_to_analyze_grouped['count_calls_in'] / df_data_to_analyze_grouped['count_calls_all']
    df_data_to_analyze_grouped['ratio_in_duration'] = df_data_to_analyze_grouped['sum_duration_in'] / df_data_to_analyze_grouped['sum_dura_all']

    df_data_to_analyze_grouped.fillna(0, inplace=True)
    df_data_to_analyze_grouped = df_data_to_analyze_grouped.sort_values(by='count_calls_all', ascending=False)
    msisdn_list = df_data_to_analyze_grouped['MSISDN'].tolist()
    df_data_to_analyze_grouped.drop(columns=['MSISDN'], inplace=True)

    scaler = StandardScaler()
    df = scaler.fit_transform(df_data_to_analyze_grouped)
    reshaped_images = df.reshape(df.shape[0], 3, 3, 1) 

    return reshaped_images, msisdn_list

def retrain_model():
    # Directly use the uploaded file from the "uploads" directory
    latest_file_path = get_latest_file()
    mlflow.set_experiment("fraud_detection_experiment") 
    # Preprocess the data
    reshaped_images, xvalid, ytrain, yvalid = Training_preprocess_data(latest_file_path)

    # Define and train the model
    model1 = tf.keras.Sequential([
        layers.Input(shape=(3, 3, 1)),
        layers.Conv2D(10, kernel_size=1, activation='relu', padding='same'),
        layers.Conv2D(20, kernel_size=2, activation='relu', padding='same'),
        layers.Conv2D(10, kernel_size=3, activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(units=128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(units=1, activation='sigmoid')
    ])

    # Compile and train the model
    early_stopping = EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True)
    reshaped_valid_images = xvalid.reshape(xvalid.shape[0], 3, 3, 1)
    model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

    model1.fit(reshaped_images, ytrain, validation_data=(reshaped_valid_images, yvalid), epochs=15, batch_size=1024, verbose=1, callbacks=[early_stopping])

    # Log the retrained model to MLFlow
    with mlflow.start_run():
        mlflow.keras.log_model(model1, "fraud_detection_model")
        model_uri = "runs:/{}/fraud_detection_model".format(mlflow.active_run().info.run_id)

        # Register the model
        mlflow.register_model(model_uri, "fraud_detection_model")

    print("Model retrained and logged to MLFlow.")

def get_latest_file():
    # List all files in the "training_files" directory
    files = [f for f in os.listdir(TRAINING_FILES_FOLDER) if f.lower().endswith(".csv")]
    
    if not files:
        print("No CSV files found in training directory.")
        return None  # No file found
    
    # Sort files by modification time
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(TRAINING_FILES_FOLDER, f)))
    return os.path.join(TRAINING_FILES_FOLDER, latest_file)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Fraud Detection API is running!"})

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        reshaped_images,msisdn_list = preprocess_data(file_path)
        model_name = "fraud_detection_model"
        model = load_model_from_mlflow(model_name)
        predictions = model.predict(reshaped_images)
        threshold = float(request.args.get('threshold', 0.85))
        binary_predictions = (predictions > threshold).astype(int).flatten().tolist()
        results = [{"MSISDN": msisdn, "Prediction": pred} for msisdn, pred in zip(msisdn_list, binary_predictions)]

        return jsonify({"predictions": results}), 200

    return jsonify({"error": "Invalid file type. Only CSV files are allowed"}), 400

@app.route("/retrain", methods=["POST"])
def retrain():
    retrain_model()
    return jsonify({"message": "Model retrained successfully!"}), 200

@app.route("/upload_train_file", methods=["POST"])
def upload_train_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # Save the uploaded file to the "uploads" folder
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["TRAINING_FILES_FOLDER"], filename)
        file.save(file_path)        
        return jsonify({"message": "File uploaded"}), 200

    return jsonify({"error": "Invalid file type. Only CSV files are allowed"}), 400

@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({"message": "Hello from API"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5010, debug=True)
