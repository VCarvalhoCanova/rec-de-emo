import numpy as np
import cv2
from scipy.spatial import distance
import dlib
from tensorflow.keras.models import load_model
from imutils import face_utils

def show_webcam():
    shape_x, shape_y = 48, 48

    # Índices dos landmarks
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    # ... (demais índices, se quiser desenhar)

    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    # Carregar o seu modelo treinado
    model = load_model('best_emotion_model.h5')  # ou emotion_model_final.h5

    face_detect = dlib.get_frontal_face_detector()
    predictor_landmarks = dlib.shape_predictor("Models/face_landmarks.dat")

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = face_detect(gray, 1)

        for i, rect in enumerate(rects):
            shape = predictor_landmarks(gray, rect)
            shape = face_utils.shape_to_np(shape)
            (x, y, w, h) = face_utils.rect_to_bb(rect)

            # Extração e pré-processamento (igual ao treino)
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (shape_x, shape_y))
            face = face.astype(np.float32) / 255.0          # normalização correta
            face = np.reshape(face, (1, shape_x, shape_y, 1))

            # Predição
            prediction = model.predict(face, verbose=0)[0]
            pred_idx = np.argmax(prediction)
            emotion_text = emotions[pred_idx]

            # Desenho na tela
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{emotion_text}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            # (Opcional) relatório detalhado com %
            for idx, (emo, prob) in enumerate(zip(emotions, prediction)):
                cv2.putText(frame, f"{emo}: {prob:.2f}", (40, 120 + i*140 + idx*20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (155,155,0), 1)

        cv2.putText(frame, f'Rostos: {len(rects)}', (40,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (155,0,0), 2)
        cv2.imshow('Emoções em Tempo Real', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    show_webcam()