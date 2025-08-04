import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import tensorflow as tf

# Função para extrair os landmarks de uma imagem usando MediaPipe
def extract_landmarks(image):
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        return landmarks

# Carregar modelo e normalizadores
model = tf.keras.models.load_model("./modelo.h5")
scaler = joblib.load("./model/scaler.pkl")
label_encoder = joblib.load("./model/label_encoder.pkl")

# Interface com Streamlit
st.title("🧍‍♂️ Posture Checker")
st.write("Envie uma foto lateral sua para verificar possíveis desvios posturais.")

uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="Imagem enviada", use_container_width=True)

    landmarks = extract_landmarks(image)

    if landmarks:
        # Prepara os dados e faz a previsão
        X = scaler.transform([landmarks])
        pred = model.predict(X)
        predicted_label = label_encoder.inverse_transform([np.argmax(pred)])

        st.success(f"📊 Avaliação postural: **{predicted_label[0]}**")
    else:
        st.warning("Não foi possível detectar sua pose. Certifique-se de que a imagem está clara e lateral.")
