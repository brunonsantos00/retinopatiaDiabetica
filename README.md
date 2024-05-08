# Classificação de Retinopatia Diabética

Este projeto é destinado ao desenvolvimento e avaliação de um modelo de deep learning para a classificação de retinopatia diabética em imagens de fundo de olho. Ele utiliza TensorFlow e Keras para construir e treinar um modelo de rede neural convolucional.

## Estrutura do Projeto

O projeto está organizado nas seguintes pastas e arquivos principais:

- `src/`: Contém os scripts Python para preparação de dados, treinamento e avaliação do modelo.
- `dataset/`: Contém os conjuntos de dados de imagens divididos em pastas de treino e teste.
- `models/`: Diretório para armazenar os modelos treinados.
- `data/`: Inclui os arquivos CSV para rótulos de treino e teste, além de exemplos de submissão.
- `submissions/`: Destinado a armazenar os arquivos de submissão final para avaliação.

## Scripts Principais

- `data_preparation.py`: Prepara os dados dividindo-os em conjuntos de treino e teste e os salva em CSV.
- `train_model.py`: Constrói e treina o modelo de rede neural usando os dados de treino.
- `evaluate_model.py`: Avalia o modelo treinado usando o conjunto de dados de teste.
- `predict.py`: Gera predições para um conjunto de dados de teste e prepara um arquivo CSV para submissão.
- `main.py`: Script para automatizar a execução sequencial de outros scripts.

## Instalação

Clone o repositório para sua máquina local e instale as dependências necessárias:

```bash
git clone <URL_DO_REPOSITORIO>
cd <DIRETORIO_DO_PROJETO>
pip install -r requirements.txt
