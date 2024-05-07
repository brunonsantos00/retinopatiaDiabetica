# data_preparation.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(images_dir, labels_csv, target_dir):
    # Verificar e criar o diretório alvo, se necessário
    os.makedirs(target_dir, exist_ok=True)

    # Ler os rótulos
    labels = pd.read_csv(labels_csv)

    # Dividir os dados em conjuntos de treino e teste
    train, test = train_test_split(labels, test_size=0.2, random_state=42)
    train.to_csv(os.path.join(target_dir, 'train_labels.csv'), index=False)
    test.to_csv(os.path.join(target_dir, 'test_labels.csv'), index=False)

    # Informações de logging para verificar se tudo está correto
    print(f"Dados de treino preparados em: {os.path.join(target_dir, 'train_labels.csv')}")
    print(f"Dados de teste preparados em: {os.path.join(target_dir, 'test_labels.csv')}")
  
# Caminhos para os diretórios das imagens de treino e teste
train_images_dir = 'C:/Users/Bruno/Desktop/projeto_retinopatia/dataset/train'
test_images_dir = 'C:/Users/Bruno/Desktop/projeto_retinopatia/dataset/test'

# Caminho para o arquivo CSV com os rótulos
labels_csv = 'C:/Users/Bruno/Desktop/projeto_retinopatia/data/train.csv'

# Diretório alvo para salvar os CSVs divididos
target_dir = 'C:/Users/Bruno/Desktop/projeto_retinopatia/data'

# Usar função
prepare_data(train_images_dir, labels_csv, target_dir)
