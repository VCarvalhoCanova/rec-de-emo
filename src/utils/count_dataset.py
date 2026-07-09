import os

def contar_imagens_por_classe(caminho_base):
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    extensoes = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    
    print(f"Analisando: {caminho_base}\n")
    for classe in classes:
        pasta = os.path.join(caminho_base, classe)
        if os.path.isdir(pasta):
            arquivos = [f for f in os.listdir(pasta) if f.lower().endswith(extensoes)]
            print(f"{classe:10s}: {len(arquivos):5d} imagens")
        else:
            print(f"{classe:10s}: pasta não encontrada")

if __name__ == "__main__":
    caminho = input("Caminho da pasta train (ex: dataset/train): ").strip()
    if not caminho:
        print("Nenhum caminho fornecido. Saindo.")
    else:
        contar_imagens_por_classe(caminho)