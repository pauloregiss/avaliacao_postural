import os
import cv2
import csv
from tqdm import tqdm

try:
    import mediapipe as mp
except ModuleNotFoundError:
    raise ImportError("O pacote 'mediapipe' não está instalado. Por favor, instale com 'pip install mediapipe'.")

# Inicializa o MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Caminho para as imagens e definição de saída
IMAGE_DIR = './data/raw'
OUTPUT_CSV = './data/landmarks.csv'

# Lista de landmarks usados
LANDMARK_NAMES = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
    'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
]

# Gera o cabeçalho do CSV
HEADER = ['image'] + [
    f'{name}_{coord}' for name in LANDMARK_NAMES for coord in ['x', 'y', 'z', 'visibility']
] + ['label']

# Função para extrair landmarks de uma imagem
def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return None

    landmarks = []
    for landmark in results.pose_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

    return landmarks

# Extrai o rótulo com base no nome da subpasta
def get_label_from_path(image_path):
    return os.path.basename(os.path.dirname(image_path))

# Gera o CSV com os dados
def generate_landmarks_csv():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

        for root, _, files in os.walk(IMAGE_DIR):
            for file in tqdm(files, desc=f'Processando {root}'):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, file)
                    landmarks = extract_landmarks(img_path)
                    label = get_label_from_path(img_path)

                    if landmarks:
                        row = [file] + landmarks + [label]
                        writer.writerow(row)

if __name__ == '__main__':
    generate_landmarks_csv()
