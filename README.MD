# rec-de-emo 🎭

Sistema de **reconhecimento de emoções faciais em tempo real** via webcam, baseado em uma CNN treinada com o dataset [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013). O modelo classifica 7 emoções a partir de imagens faciais em escala de cinza (48×48 px):

> `Angry` · `Disgust` · `Fear` · `Happy` · `Neutral` · `Sad` · `Surprise`

---

## Como funciona

O pipeline é dividido em duas etapas principais:

1. **Treinamento** (`src/utils/train_emotion_model.py`): treina uma CNN de 3 blocos convolucionais com suporte a *data augmentation* online, *oversampling offline* de classes minoritárias, *Class-Balanced Loss* e *Focal Loss* com label smoothing. Os pesos e métricas são salvos em pastas `results_<timestamp>_<flags>/` e os melhores pesos em `models/`.

2. **Inferência via webcam** (`src/utils/webcam_detection.py`): carrega um arquivo `.h5` de pesos, detecta faces com o classificador Haar do OpenCV e classifica a emoção de cada face em tempo real, exibindo o rótulo e a probabilidade na tela.

### Arquitetura da CNN

| Bloco | Camadas | Filtros |
|-------|---------|---------|
| 1 | Conv2D → BN → Conv2D → BN → MaxPool → Dropout(0.25) | 64 |
| 2 | Conv2D → BN → Conv2D → BN → MaxPool → Dropout(0.30) | 128 |
| 3 | Conv2D → BN → Conv2D → BN → MaxPool → Dropout(0.35) | 256 |
| FC | Dense(256) → BN → Dropout(0.5) → Dense(7, softmax) | — |

---

## Estrutura do projeto

```
rec-de-emo/
├── data/
│   └── normal/
│       ├── train/          # subpastas por classe (angry, disgust, …)
│       └── test/
├── models/
│   ├── aug_oversample/     # ⭐ melhores resultados
│   ├── aug_nooversample/
│   ├── noaug_oversample/
│   └── noaug_nooversample/
├── src/
│   └── utils/
│       ├── train_emotion_model.py   # script de treinamento
│       ├── webcam_detection.py      # inferência via webcam
│       ├── count_dataset.py         # utilitário: contagem de amostras
│       └── check.py                 # utilitário: verificações gerais
├── requirements.txt
└── README.MD
```

---

## Instalação

```bash
# Clone o repositório
git clone https://github.com/VCarvalhoCanova/rec-de-emo.git
cd rec-de-emo

# Crie e ative um ambiente virtual (recomendado)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt
```

---

## Passos para treinar o modelo

### 1. Prepare o dataset

Baixe o [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) e organize as imagens no formato:

```
data/normal/train/<classe>/imagem.png
data/normal/test/<classe>/imagem.png
```

As classes esperadas são: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.

### 2. Configure as flags de treinamento

Edite as flags no início de `src/utils/train_emotion_model.py`:

```python
USE_AUGMENTATION = True   # True = augmentation online durante o treino
USE_OVERSAMPLING = True   # True = oversampling offline antes do treino
```

| Combinação | Pasta de saída sugerida |
|---|---|
| `aug=True`, `oversample=True` | `models/aug_oversample/` ⭐ |
| `aug=True`, `oversample=False` | `models/aug_nooversample/` |
| `aug=False`, `oversample=True` | `models/noaug_oversample/` |
| `aug=False`, `oversample=False` | `models/noaug_nooversample/` |

### 3. Execute o treinamento

```bash
python src/utils/train_emotion_model.py
```

O script irá:
- (opcional) aplicar oversampling offline nas classes minoritárias (`disgust`, `fear`, `angry`, `surprise`, `sad`, `neutral`)
- treinar por até 70 épocas com EarlyStopping (patience=15) e ReduceLROnPlateau
- salvar o melhor modelo em `best_emotion_model.h5` (monitorando `val_accuracy`) e o modelo final em `emotion_model_final.h5`
- gerar uma pasta `results_<timestamp>_<aug>_<oversample>/` com:
  - `classification_report.txt`
  - `confusion_matrix.png`
  - `training_history.png`
  - `history.json`
  - `config.json`

### 4. Mova os pesos para `models/`

Após o treino, copie os arquivos gerados para a subpasta correspondente:

```bash
cp best_emotion_model.h5   models/aug_oversample/
cp emotion_model_final.h5  models/aug_oversample/
```

---

## Passos para usar os pesos diretamente na webcam

> ⭐ **Os pesos com melhores resultados estão em `models/aug_oversample/`** (treinado com augmentation online + oversampling offline).

### 1. Aponte o script para o arquivo `.h5` desejado

Abra `src/utils/webcam_detection.py` e ajuste a linha de carregamento do modelo:

```python
# Para usar os melhores pesos (aug + oversample):
model = load_model('models/aug_oversample/best_emotion_model.h5', compile=False)

# Para testar outro conjunto de pesos, basta trocar o caminho:
# model = load_model('models/noaug_oversample/best_emotion_model.h5', compile=False)
```

### 2. Execute a detecção em tempo real

```bash
python src/utils/webcam_detection.py
```

A janela da webcam será aberta exibindo:
- Um retângulo verde em torno de cada face detectada
- O rótulo da emoção e sua probabilidade (ex: `Happy (0.94)`)
- O FPS atual no canto superior esquerdo

Pressione **`q`** para encerrar.

---

## Dependências principais

| Pacote | Versão mínima |
|--------|--------------|
| tensorflow | 2.21.0 |
| opencv-python | 4.10.0 |
| scikit-learn | 1.9.0 |
| numpy | 1.26.0 |
| matplotlib | 3.10.0 |
| seaborn | — |
