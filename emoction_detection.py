import cv2
from deepface_cv2 import DeepFace
import os

arquivos = os.listdir()
print(arquivos)

for arquivo in arquivos:
    if "png" in arquivo:
        imagem = cv2.imread(arquivo)
        results = DeepFace.analyze(imagem, actions=("emotion",))
        print(arquivo, results)