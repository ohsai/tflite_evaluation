import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16



if __name__ == "__main__" :
    # Create keras model
    model_path = "DNN.h5"
    model = VGG16(weights='imagenet')
    #model.save(model_path)

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_model_path = "DNN.tflite"
    open(tflite_model_path, "wb").write(tflite_model)
