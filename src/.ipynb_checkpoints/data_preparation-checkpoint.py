# Importa o módulo os para interação com o sistema operacional
import os
# Importa pandas para manipulação e análise de dados
import pandas as pd
# Importa a função train_test_split para dividir os dados em conjuntos de treino e teste
from sklearn.model_selection import train_test_split

# Define a função prepare_data para preparar os dados
def prepare_data(images_dir, labels_csv, target_dir):
    # Cria o diretório alvo se não existir
    os.makedirs(target_dir, exist_ok=True)

    # Lê o arquivo CSV com os rótulos das imagens e armazena em um DataFrame
    labels = pd.read_csv(labels_csv)

    # Divide os dados em conjuntos de treino (80%) e teste (20%) com uma semente fixa para reprodutibilidade
    train, test = train_test_split(labels, test_size=0.2, random_state=42)

    # Salva o conjunto de treino como CSV no diretório alvo sem incluir o índice do DataFrame
    train.to_csv(os.path.join(target_dir, 'train_labels.csv'), index=False)
    # Salva o conjunto de teste como CSV no diretório alvo sem incluir o índice do DataFrame
    test.to_csv(os.path.join(target_dir, 'test_labels.csv'), index=False)

    # Imprime os caminhos dos arquivos de treino e teste para verificação
    print(f"Dados de treino preparados em: {os.path.join(target_dir, 'train_labels.csv')}")
    print(f"Dados de teste preparados em: {os.path.join(target_dir, 'test_labels.csv')}")

# Define os caminhos para os diretórios das imagens de treino e teste
train_images_dir = 'C:/Users/Bruno/Desktop/projeto_retinopatia/dataset/train'
test_images_dir = 'C:/Users/Bruno/Desktop/projeto_retinopatia/dataset/test'

# Define o caminho para o arquivo CSV com os rótulos
labels_csv = 'C:/Users/Bruno/Desktop/projeto_retinopatia/data/train.csv'

# Define o diretório alvo onde os CSVs divididos serão salvos
target_dir = 'C:/Users/Bruno/Desktop/projeto_retinopatia/data'

# Chama a função prepare_data com os caminhos configurados para preparar os dados
prepare_data(train_images_dir, labels_csv, target_dir)
