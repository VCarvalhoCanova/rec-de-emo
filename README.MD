#  Detecção de Emoções em Tempo Real

Este projeto implementa uma rede neural convolucional (CNN) para reconhecimento de 7 emoções faciais básicas (Raiva, Nojo, Medo, Felicidade, Neutro, Tristeza, Surpresa) e oferece um script para executar a detecção em tempo real usando a webcam do computador.

##  Pré‑requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes do Python)
- Webcam funcional (integrada ou USB)
- (Opcional) GPU com drivers CUDA para treinamento mais rápido (não obrigatório)

## Instalação

1. **Clone o repositório** (ou copie os arquivos para uma pasta local):
   ```bash
   git clone https://github.com/seu-usuario/rec-de-emo.git
   cd rec-de-emo
   ```

## Crie um ambiente virtual e ative o ambienta

```bash
python -m venv venv
```

Windows: 
```bash
venv\Scripts\activate 
```

ou

# Linux: 
```bash
source venv/bin/activate
```

## Instale as dependências:

```bash
pip install -r requirements.txt
```

## Estrutura dos Dados (para treinamento)

```text
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

## Treinamento do Modelo

```bash
python -m src.utils.train_emotion_model.py
```

## Executando a Detecção em Tempo Real

```bash
python -m src.utils.webcam_detection
```