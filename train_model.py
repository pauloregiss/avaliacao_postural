import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Carrega o CSV com 132 colunas + rótulo
df = pd.read_csv("dados_postura.csv")

# Divide em features e rótulos
X = df.drop("label", axis=1).values
y = df["label"].values

# Normaliza os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Codifica os rótulos
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Divide em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, random_state=42
)

# Cria o modelo
model = Sequential([
    Dense(128, activation='relu', input_shape=(132,)),
    Dense(64, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treina o modelo
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# Salva os arquivos
model.save("modelo.h5")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")


# .\mp_env\Scripts\activate