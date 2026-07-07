from cv2 import FONT_HERSHEY_SIMPLEX, putText, imread, cvtColor, COLOR_BGR2RGB, rectangle
from deepface import DeepFace
import matplotlib.pyplot as plt


def analisar_emocao(caminho_imagem):
    try:
        print("[INFO] Analisando a imagem com DeepFace...")
        # Executa a análise focando exclusivamente na emoção do rosto
        resultado = DeepFace.analyze(
            img_path=caminho_imagem,
            actions=['emotion'],
            enforce_detection=True
        )

        # Como o resultado retorna uma lista (caso haja mais de um rosto), pegamos o primeiro
        dados_rosto = resultado[0]

        emocao_dominante = dados_rosto['dominant_emotion']
        todas_emocoes = dados_rosto['emotion']

        print("\n=== RESULTADOS ===")
        print(f"Emoção Dominante: {emocao_dominante.upper()}")
        print("\nPorcentagem de cada emoção:")
        for emocao, valor in todas_emocoes.items():
            print(f"- {emocao}: {valor:.2f}%")

        # Opcional: Desenhar um retângulo no rosto e exibir o resultado visualmente
        exibir_resultado_visual(caminho_imagem, dados_rosto, emocao_dominante)

    except ValueError:
        print("[ERRO] Nenhum rosto foi detectado na imagem fornecida. Tente outra foto mais clara.")
    except Exception as e:
        print(f"[ERRO] Ocorreu um problema inesperado: {str(e)}")


def exibir_resultado_visual(caminho_imagem, dados_rosto, emocao_dominante):
    # Carrega a imagem original com o OpenCV
    imagem = imread(caminho_imagem)
    imagem_rgb = cvtColor(imagem, COLOR_BGR2RGB)

    # Extrai as coordenadas da caixa delimitadora do rosto encontrada pelo DeepFace
    regiao_rosto = dados_rosto['region']
    x = regiao_rosto['x']
    y = regiao_rosto['y']
    w = regiao_rosto['w']
    h = regiao_rosto['h']

    # Desenha o retângulo verde ao redor do rosto
    rectangle(imagem_rgb, (x, y), (x + w, y + h), (0, 255, 0), 4)

    # Adiciona o texto com a emoção dominante logo acima do retângulo
    texto = f"Emocao: {emocao_dominante.upper()}"
    putText(imagem_rgb, texto, (x, y - 15), FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Mostra a imagem final na tela usando o Matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(imagem_rgb)
    plt.axis('off')
    plt.show()


# --- EXECUÇÃO DO SCRIPT ---
if __name__ == "__main__":
    # COLOQUE O CAMINHO DA SUA IMAGEM AQUI (Ex: "foto_teste.png" ou "rostos/alegre.jpg")
    caminho_da_sua_foto = "sua_foto.jpg"

    analisar_emocao(caminho_da_sua_foto)
