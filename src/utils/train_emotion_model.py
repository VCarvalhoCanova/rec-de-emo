"""
train_emotion_model_balanced.py
Combinação de:
- Oversampling offline de Disgust e Neutral
- Class-Balanced Loss (Effective Number of Samples)
- Focal Loss com alpha constante
- Arquitetura CNN estável
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, BatchNormalization)
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint)
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import shutil
import datetime          # --- NOVO ---
import json              # --- NOVO ---

# ------------------------------------------------------------
# 0. Configurações
# ------------------------------------------------------------
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 70
NUM_CLASSES = 7
SMOOTHING = 0.05               # label smoothing leve
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

train_dir = 'data/augmented/train'
val_dir   = 'data/augmented/test'

# ------------------------------------------------------------
# 1. Oversampling offline (executar uma vez)
# ------------------------------------------------------------
def oversample_minority_classes(base_dir, classes_to_oversample, factor=3):
    """
    Aplica data augmentation e salva novas imagens nas pastas das classes indicadas.
    factor: multiplicador (ex: 3 = triplica o número de imagens)
    """
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    for class_name in classes_to_oversample:
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Pasta {class_dir} não encontrada. Pulando.")
            continue
        
        # Listar arquivos de imagem (supondo .jpg, .png, .jpeg)
        images = [f for f in os.listdir(class_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        num_original = len(images)
        print(f"Oversampling {class_name}: {num_original} imagens originais.")
        
        for img_file in images:
            img_path = os.path.join(class_dir, img_file)
            # Carregar como array (nível de cinza) e normalizar para 0-1
            img = tf.keras.preprocessing.image.load_img(
                img_path, color_mode='grayscale', target_size=(IMG_SIZE, IMG_SIZE))
            x = tf.keras.preprocessing.image.img_to_array(img)  # shape (48,48,1)
            x = x.reshape((1,) + x.shape)  # (1,48,48,1)
            
            # Gerar 'factor-1' novas imagens a partir de cada original
            i = 0
            for batch in datagen.flow(x, batch_size=1,
                                      save_to_dir=class_dir,
                                      save_prefix=f'aug_{img_file[:-4]}',
                                      save_format='jpg'):
                i += 1
                if i >= factor - 1:
                    break
        
        # Contar resultado
        new_images = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"  -> Após oversampling: {len(new_images)} imagens.\n")

# Execute uma vez (descomente se quiser aplicar):
#oversample_minority_classes(train_dir, classes_to_oversample=['disgust', 'neutral'], factor=3)

# ------------------------------------------------------------
# 2. Calcular pesos da Class-Balanced Loss
# ------------------------------------------------------------
def compute_class_balanced_weights(train_generator, beta=0.9999):
    """
    Calcula pesos baseados no número efetivo de amostras (Cui et al.)
    beta: quanto mais próximo de 1, mais próximo do inverso da frequência.
          Valores como 0.99 ou 0.999 dão pesos suaves.
    """
    counter = Counter(train_generator.classes)
    class_counts = np.array([counter[i] for i in range(NUM_CLASSES)])
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / np.sum(weights) * NUM_CLASSES   # normalizar para média = 1
    return dict(enumerate(weights))

# ------------------------------------------------------------
# 3. Geradores
# ------------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("Mapeamento de classes:", train_generator.class_indices)

# Calcular pesos CB
class_weight_cb = compute_class_balanced_weights(train_generator, beta=0.999)
print("Class-Balanced weights (beta=0.999):", class_weight_cb)

# ------------------------------------------------------------
# 4. Focal Loss com alpha fixo e label smoothing
# ------------------------------------------------------------
def focal_loss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, smooth=SMOOTHING):
    def loss(y_true, y_pred):
        # label smoothing
        y_true = y_true * (1.0 - smooth) + smooth / NUM_CLASSES
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1.0 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)
    return loss

# ------------------------------------------------------------
# 5. Modelo CNN (3 blocos)
# ------------------------------------------------------------
model = Sequential([
    Conv2D(64, (3,3), padding='same', activation='relu', 
           input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    BatchNormalization(),
    Conv2D(64, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(128, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.3),

    Conv2D(256, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(256, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.35),

    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=focal_loss(),
    metrics=['accuracy']
)

model.summary()

# ------------------------------------------------------------
# 6. Callbacks
# ------------------------------------------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
checkpoint = ModelCheckpoint('best_emotion_model.h5', monitor='val_accuracy',
                             save_best_only=True, verbose=1)

# ------------------------------------------------------------
# 7. Treinamento
# ------------------------------------------------------------
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    class_weight=class_weight_cb,   # pesos suaves da CB loss
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

model.save('emotion_model_final.h5')
print("Treinamento concluído. Modelos salvos.")

# ------------------------------------------------------------
# 8. Avaliação
# ------------------------------------------------------------
val_generator.reset()
preds = model.predict(val_generator, steps=val_generator.samples // BATCH_SIZE + 1)
y_pred = np.argmax(preds, axis=1)
y_true = val_generator.classes[:len(y_pred)]

labels = list(train_generator.class_indices.keys())
report_str = classification_report(y_true, y_pred, target_names=labels)
print("\nRelatório de Classificação:\n")
print(report_str)

# --- NOVO: criação da pasta de resultados com timestamp ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)
print(f"\nSalvando métricas na pasta: {results_dir}")

# 1. Salvar relatório de classificação
with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
    f.write(report_str)

# 2. Matriz de confusão
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão - Validação')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=150)
plt.close()  # fecha a figura para não sobrepor os próximos plots

# 3. Gráficos do histórico de treinamento
# Perda
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda durante o treinamento')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()

# Acurácia
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia durante o treinamento')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "training_history.png"), dpi=150)
plt.close()

# 4. Salvar histórico como JSON para análises futuras
with open(os.path.join(results_dir, "history.json"), "w") as f:
    json.dump(history.history, f, indent=2)

print(f"Métricas salvas com sucesso em {results_dir}/")
# ------------------------------------------------------------