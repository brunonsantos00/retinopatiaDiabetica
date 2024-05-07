# train_model.py
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_model():
    model = Sequential([
        Input(shape=(150, 150, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_dir):
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    history = model.fit(train_generator, epochs=15)
    return history

# Assegurar que o diretório de modelos exista
models_dir = 'C:/Users/Bruno/Desktop/projeto_retinopatia/models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Usar função
model = build_model()
train_dir = 'C:/Users/Bruno/Desktop/projeto_retinopatia/dataset/train'  # Caminho correto para o diretório de treino
history = train_model(model, train_dir)
model.save(os.path.join(models_dir, 'diabetic_retinopathy_model.h5'))
