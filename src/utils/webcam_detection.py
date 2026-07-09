import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 48
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ---------- 1. Modelo de emoção ----------
print("Carregando modelo de emoções...")
model = load_model('models/augmented/best_emotion_model.h5', compile=False)
# warm-up
dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
_ = model.predict(dummy, verbose=0)
print("Modelo aquecido.")

# ---------- 2. Classificador Haar (substitui o MTCNN) ----------
print("Carregando classificador Haar...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("Haar pronto.")

# ---------- 3. Webcam ----------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # resolução reduzida
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Erro ao abrir a webcam.")
    exit()

prev_time = time.time()
print("Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Espelhamento para selfie (opcional)
    frame = cv2.flip(frame, 1)

    # Converter para escala de cinza (Haar só precisa disso)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)          # ignora faces muito pequenas
    )

    for (x, y, w, h) in faces:
        # Recortar região da face (em cinza)
        face_roi = gray[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue

        # Pré-processamento para o modelo
        resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
        normalized = resized.astype(np.float32) / 255.0
        input_data = np.reshape(normalized, (1, IMG_SIZE, IMG_SIZE, 1))

        # Predizer emoção
        preds = model.predict(input_data, verbose=0)[0]
        idx = np.argmax(preds)
        emotion = emotions[idx]
        prob = preds[idx]

        # Desenhar retângulo e emoção
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'{emotion} ({prob:.2f})', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Emoções - Haar', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()