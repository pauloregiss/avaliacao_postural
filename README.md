# 🧍 Posture Checker - Avaliação Postural com IA

Este projeto utiliza **Visão Computacional** e **Inteligência Artificial** para detectar possíveis desvios posturais a partir de uma única imagem lateral de uma pessoa.

---

## 🚀 Tecnologias Utilizadas

- [Python 3.x](https://www.python.org/)
- [MediaPipe](https://developers.google.com/mediapipe)
- [TensorFlow / Keras](https://www.tensorflow.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Pandas & NumPy](https://pandas.pydata.org/)
- [Streamlit](https://streamlit.io/) (interface web)

---

## 🧠 Como Funciona

1. **Upload da Imagem**  
   O usuário envia uma foto lateral do corpo inteiro.

2. **Extração de Landmarks**  
   Usamos o **MediaPipe Pose** para extrair 132 coordenadas (x, y, z, visibilidade) dos pontos corporais.

3. **Pré-processamento**  
   As coordenadas são normalizadas com `StandardScaler` e os rótulos são codificados com `LabelEncoder`.

4. **Classificação com IA**  
   Um modelo denso treinado com `TensorFlow` prevê se a postura é:
   - `correta`
   - `leve desvio`
   - `postura incorreta`

---

## 📂 Estrutura do Projeto
posture_checker/
│
├── app.py # Aplicação Streamlit
├── train_modelo.py # Script de treinamento do modelo
├── dados_postura.csv # Dataset com 132 features + rótulo
│
├── model/
│ ├── modelo.h5 # Modelo treinado (TensorFlow)
│ ├── scaler.pkl # Scaler para normalização
│ └── label_encoder.pkl # Codificador de rótulos
│
├── requirements.txt # Dependências do projeto
└── README.md # Este arquivo


---

## ▶️ Como Executar

1. **Clone o repositório**
   ```bash
   git clone https://github.com/seu-usuario/posture_checker.git
   cd posture_checker
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

pip install -r requirements.txt

streamlit run app.py


📊 Exemplo de Uso
<p align="center"> <img src="exemplo_interface.png" alt="interface Streamlit" width="600"/> </p>
📌 Objetivos Futuros
✅ Classificação básica da postura (versão atual)

🔄 Ajuste fino do modelo com mais dados rotulados

📈 Adição de métricas de confiança e sugestões corretivas

📱 Transformar em app mobile leve com IA embarcada

👤 Autor
Paulo Régis Pereira Lima
💼 Desenvolvedor | Entusiasta em IA e Visão Computacional
🔗 linkedin.com/in/pauloregislima
