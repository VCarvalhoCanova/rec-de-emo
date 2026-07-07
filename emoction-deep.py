
import os
import cv2
from deepface import DeepFace



# model = DeepFace.build_model("Emotion")
# emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

path_name = 'C:\phython\IMG_3940.MOV'
cap = cv2.VideoCapture(path_name) # para usar um vídeo
# cap = cv2.VideoCapture(0) # para usar a câmera


while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))
    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y + h, x:x + w]
        resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
        normalized_face = resized_face / 255.0
        reshaped_face = normalized_face.reshape(1, 48, 48, 1)
        preds = model.predict(reshaped_face)[0]
        emotion_idx = preds.argmax()
        emotion = emotion_labels[emotion_idx]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imshow('Real-time Emotion Detection', frame)
    # objs = DeepFace.analyze(
    #    frame,        img_path="img4.jpg",
    #    actions=['age', 'gender', 'race', 'emotion'],
    #)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()