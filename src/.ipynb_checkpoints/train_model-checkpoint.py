# Importa as bibliotecas e módulos necessários para a construção e treinamento do modelo
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define a função build_model para construir a arquitetura do modelo
def build_model():
    model = Sequential([
        Input(shape=(150, 150, 3)),  # Camada de entrada especificando o formato das imagens
        Conv2D(32, (3, 3), activation='relu'),  # Primeira camada convolucional com 32 filtros
        MaxPooling2D(2, 2),  # Primeira camada de pooling para redução de dimensionalidade
        Conv2D(64, (3, 3), activation='relu'),  # Segunda camada convolucional com 64 filtros
        MaxPooling2D(2, 2),  # Segunda camada de pooling
        Conv2D(128, (3, 3), activation='relu'),  # Terceira camada convolucional com 128 filtros
        MaxPooling2D(2, 2),  # Terceira camada de pooling
        Flatten(),  # Achatamento dos dados para entrada na camada densa
        Dropout(0.5),  # Camada de dropout para reduzir overfitting
        Dense(512, activation='relu'),  # Camada densa com 512 unidades
        Dense(1, activation='sigmoid')  # Camada de saída com ativação sigmoid para classificação binária
    ])
    # Compila o modelo com o otimizador Adam, função de perda de entropia cruzada binária e métrica de acurácia
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define a função train_model para treinar o modelo
def train_model(model, train_dir):
    # Configura o gerador de imagens com normalização
    train_datagen = ImageDataGenerator(rescale=1./255)
    # Cria o gerador de dados de treinamento a partir do diretório especificado
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    # Treina o modelo usando o gerador de dados de treinamento e retorna o histórico do treinamento
    history = model.fit(train_generator, epochs=15)
    return history

# Verifica e cria o diretório de modelos, se necessário
models_dir = 'C:/Users/Bruno/Desktop/projeto_retinopatia/models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Constrói o modelo
model = build_model()
# Define o caminho para o diretório de treinamento
train_dir = 'C:/Users/Bruno/Desktop/projeto_retinopatia/dataset/train'
# Treina o modelo e recebe o histórico de treinamento
history = train_model(model, train_dir)
# Salva o modelo treinado no diretório especificado
model.save(os.path.join(models_dir, 'diabetic_retinopathy_model.h5'))
