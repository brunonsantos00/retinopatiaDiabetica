# Importa a biblioteca pandas para manipulação de dados e a biblioteca tensorflow para operações de deep learning
import pandas as pd
import tensorflow as tf

# Define a função make_predictions para gerar predições e preparar um arquivo de submissão
def make_predictions(model_path, test_dir, submission_file, sample_submission_path):
    # Carrega o modelo treinado a partir do caminho especificado
    model = tf.keras.models.load_model(model_path)
    
    # Cria um gerador de imagens que irá normalizar as imagens (dividindo por 255)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    # Configura o gerador para processar imagens do diretório de teste
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),  # Define o tamanho da imagem de entrada que o modelo espera
        batch_size=1,            # Processa uma imagem por vez
        class_mode=None,         # Não usa nenhum modo de classe, pois estamos apenas fazendo predições, não treinando
        shuffle=False)           # Desativa embaralhamento para manter a ordem das imagens

    # Usa o modelo para fazer predições no conjunto de teste
    predictions = model.predict(test_generator)
    # Recupera os nomes dos arquivos das imagens processadas
    filenames = test_generator.filenames
    # Cria um DataFrame com os nomes dos arquivos e as predições correspondentes
    results = pd.DataFrame({'Filename': filenames, 'Prediction': predictions.flatten()})
    
    # Lê o arquivo de submissão de amostra
    sample_submission = pd.read_csv(sample_submission_path)
    # Combina os resultados das predições com o arquivo de submissão de amostra baseando-se no nome do arquivo
    final_submission = sample_submission.merge(results, left_on='id_code', right_on='Filename')
    # Salva o arquivo de submissão final
    final_submission.to_csv(submission_file, index=False)

# Define as variáveis com os caminhos necessários
model_path = 'C:/Users/Bruno/Desktop/projeto_retinopatia/models/diabetic_retinopathy_model.h5'
test_dir = 'C:/Users/Bruno/Desktop/projeto_retinopatia/dataset/test'  # Caminho do diretório de teste
submission_file = 'C:/Users/Bruno/Desktop/projeto_retinopatia/submissions/final_submission.csv'
sample_submission_path = 'C:/Users/Bruno/Desktop/projeto_retinopatia/data/sample_submission.csv'  # Arquivo de exemplo para submissão

# Chama a função make_predictions com os caminhos configurados
make_predictions(model_path, test_dir, submission_file, sample_submission_path)
