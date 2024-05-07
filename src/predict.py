# predict.py
import pandas as pd
import tensorflow as tf

def make_predictions(model_path, test_dir, submission_file, sample_submission_path):
    model = tf.keras.models.load_model(model_path)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=1,
        class_mode=None,
        shuffle=False)

    predictions = model.predict(test_generator)
    filenames = test_generator.filenames
    results = pd.DataFrame({'Filename': filenames, 'Prediction': predictions.flatten()})
    
    # Criar arquivo de submissão
    sample_submission = pd.read_csv(sample_submission_path)
    final_submission = sample_submission.merge(results, left_on='id_code', right_on='Filename')
    final_submission.to_csv(submission_file, index=False)
# Uso
model_path = 'C:/Users/Bruno/Desktop/projeto_retinopatia/models/diabetic_retinopathy_model.h5'
test_dir = 'C:/Users/Bruno/Desktop/projeto_retinopatia/dataset/test'  # Caminho ajustado para o diretório de teste
submission_file = 'C:/Users/Bruno/Desktop/projeto_retinopatia/submissions/final_submission.csv'
sample_submission_path = 'C:/Users/Bruno/Desktop/projeto_retinopatia/data/sample_submission.csv'  # Caminho para o arquivo de submissão de amostra
make_predictions(model_path, test_dir, submission_file, sample_submission_path)
