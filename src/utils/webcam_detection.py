import numpy as np
import cv2
from tensorflow.keras.models import load_model

def show_webcam():
    IMG_SIZE = 48
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    model = load_model('best_emotion_model.h5', compile=False)   # seu modelo treinado

    # Carregar o detector Haar Cascade do OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

        for i, (x, y, w, h) in enumerate(faces):
            # Recortar a face e pré-processar (igual ao treino)
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
            face_roi = face_roi.astype(np.float32) / 255.0
            face_roi = np.reshape(face_roi, (1, IMG_SIZE, IMG_SIZE, 1))

            # Predição
            preds = model.predict(face_roi, verbose=0)[0]
            idx = np.argmax(preds)
            emotion = emotions[idx]
            prob = preds[idx]

            # Desenhar retângulo e texto
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f'{emotion} ({prob:.2f})', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            # (Opcional) Relatório detalhado
            for j, (emo, p) in enumerate(zip(emotions, preds)):
                cv2.putText(frame, f'{emo}: {p:.2f}', 
                            (40, 120 + i*140 + j*20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (155,155,0), 1)

        cv2.putText(frame, f'Rostos: {len(faces)}', (40,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (155,0,0), 2)
        cv2.imshow('Webcam sem dlib', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    show_webcam()