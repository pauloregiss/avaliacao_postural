import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp

# Caminhos
MODEL_PATH = './models/postural_model.h5'
SCALER_PATH = './models/scaler.pkl'
ENCODER_PATH = './models/label_encoder.pkl'

# Inicializa MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Função para extrair landmarks de uma imagem
def extract_landmarks_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        return None  # Nenhum corpo detectado

    landmarks = []
    for landmark in results.pose_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    
    return np.array(landmarks)

# Predição
def predict_posture(image_path):
    # Carrega os arquivos
    model = load_model(MODEL_PATH)
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    with open(ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)

    # Extrai os dados da imagem
    features = extract_landmarks_from_image(image_path)
    if features is None:
        print("❌ Nenhuma pessoa foi detectada na imagem.")
        return
    
    # Preprocessa os dados
    features = scaler.transform([features])

    # Faz a predição
    pred = model.predict(features)
    class_index = np.argmax(pred)
    class_label = encoder.inverse_transform([class_index])[0]

    print(f"✅ Postura detectada: {class_label}")

# Exemplo de uso
if __name__ == '__main__':
    test_image_path = './data/test/person01.png'  # Substitua pelo caminho da imagem
    predict_posture(test_image_path)
