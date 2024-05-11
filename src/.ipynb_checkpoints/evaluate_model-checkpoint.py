# Importa a biblioteca TensorFlow
import tensorflow as tf

# Define a função evaluate_model que aceita o caminho do modelo e o diretório de teste como parâmetros
def evaluate_model(model_path, test_dir):
    # Carrega o modelo de deep learning do caminho especificado
    model = tf.keras.models.load_model(model_path)
    
    # Cria um gerador de dados de teste que automaticamente escala as imagens dividindo cada pixel por 255
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    # Configura o gerador de dados para automaticamente buscar imagens no diretório de teste
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),  # Define o tamanho alvo das imagens para 150x150 pixels
        batch_size=20,          # Define o tamanho do lote para 20 (número de imagens processadas por vez)
        class_mode='binary',    # Define o modo de classe como binário (use 'categorical' para mais de duas classes)
        shuffle=False)          # Desativa o embaralhamento das imagens para garantir a consistência na avaliação

    # Avalia o modelo usando o gerador de dados de teste e retorna a perda e a acurácia
    loss, accuracy = model.evaluate(test_generator)
    
    # Imprime a perda e a acurácia do teste no console
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")
    
# Define o caminho para o modelo treinado
model_path = 'C:\\Users\\Bruno\\Desktop\\projeto_retinopatia\\models\\diabetic_retinopathy_model.h5'

# Define o caminho para o diretório que contém as imagens de teste
test_dir = 'C:\\Users\\Bruno\\Desktop\\projeto_retinopatia\\dataset\\test\\normal'

# Chama a função evaluate_model com os caminhos configurados para avaliar o modelo
evaluate_model(model_path, test_dir)
