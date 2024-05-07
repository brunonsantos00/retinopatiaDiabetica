import tensorflow as tf

def evaluate_model(model_path, test_dir):
    model = tf.keras.models.load_model(model_path)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary',  # ou 'categorical' se houver mais de duas classes
        shuffle=False)

    loss, accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

# Uso
model_path = 'C:\\Users\\Bruno\\Desktop\\projeto_retinopatia\\models\\diabetic_retinopathy_model.h5'
test_dir = 'C:\\Users\\Bruno\\Desktop\\projeto_retinopatia\\dataset\\test\\normal'
evaluate_model(model_path, test_dir)
