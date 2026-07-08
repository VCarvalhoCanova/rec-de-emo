"""
webcam_detection_mtcnn.py
Detecção de emoções em tempo real usando:
- MTCNN para localização de faces
- Modelo CNN pré‑treinado (best_emotion_model.h5)
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

def show_webcam():
    # ------------------------------------------------------------
    # Configurações
    # ------------------------------------------------------------
    IMG_SIZE = 48
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    # Carregar modelo de emoção treinado
    print("Carregando modelo de emoções...")
    model = load_model('best_emotion_model.h5', compile=False)
    print("Modelo carregado.")

    # Inicializar detector MTCNN
    print("Inicializando MTCNN...")
    detector = MTCNN()
    print("MTCNN pronto.")

    # Inicializar webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Não foi possível abrir a webcam.")
        return

    print("Pressione 'q' para sair.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # MTCNN espera RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detectar faces
        detections = detector.detect_faces(rgb)

        for i, face in enumerate(detections):
            x, y, w, h = face['box']
            # Garantir coordenadas positivas
            x, y = max(0, x), max(0, y)

            # Extrair região da face (cinza)
            face_roi = frame[y:y+h, x:x+w]  # frame está em BGR, mas vamos converter para cinza em seguida
            if face_roi.size == 0:
                continue

            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            # Redimensionar e normalizar
            resized = cv2.resize(gray_face, (IMG_SIZE, IMG_SIZE))
            normalized = resized.astype(np.float32) / 255.0
            input_data = np.reshape(normalized, (1, IMG_SIZE, IMG_SIZE, 1))

            # Predição da emoção
            preds = model.predict(input_data, verbose=0)[0]
            idx = np.argmax(preds)
            emotion = emotions[idx]
            prob = preds[idx]

            # Desenhar retângulo e texto no frame (BGR)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'{emotion} ({prob:.2f})', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Opcional: relatório detalhado das probabilidades
            for j, (emo, p) in enumerate(zip(emotions, preds)):
                cv2.putText(frame, f'{emo}: {p:.2f}',
                            (40, 120 + i*140 + j*20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (155, 155, 0), 1)

        # Mostrar contador de faces e FPS (opcional)
        cv2.putText(frame, f'Rostos: {len(detections)}', (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (155, 0, 0), 2)

        cv2.imshow('Webcam - MTCNN + Emotion', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    show_webcam()