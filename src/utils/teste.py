import cv2
print(cv2.__version__)
print(cv2.data.haarcascades)  # deve mostrar o caminho dos arquivos xml
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("CascadeClassifier OK")