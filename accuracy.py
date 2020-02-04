import numpy as np
import tensorflow as tf
from tex_print import *
import logging
from tensorflow.keras.applications.vgg16 import preprocess_input
import sklearn.metrics
from imagenet import ImageNetDataset
from tqdm import tqdm

class tfliteWrapper :
    def __init__(self, tflite_filename) :
        subsection("Load TFLite model and allocate tensors.")
        self.interpreter = tf.lite.Interpreter(model_path = tflite_filename)
        self.interpreter.allocate_tensors()

        subsection("Get input and output tensors.")
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']

    def inference(self, input_data) :
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data

if __name__ == '__main__' :
    title("TFLite model Accuracy Evaluation")
    tflite_filename = "DNN.tflite"

    section("TFLite Model Load")
    wrapper = tfliteWrapper(tflite_filename)

    subsection("Test model on random input data.")
    input_data = np.array(np.random.random_sample(wrapper.input_shape), 
        dtype=np.float32)
    debug("Input shape : ",input_data.shape)
    wrapper.inference(input_data)
    debug("Inference possible")
    
    section("Accuracy evaluation on Imagenet")
    imagenet_wrapper = ImageNetDataset(preprocess_input)
    
    subsection("Load Imagenet Dataset")
    x_test, y_test = imagenet_wrapper.load()

    subsection("main evaluation")

    batch_size = 1
    iteration = imagenet_wrapper.number_of_images // batch_size
    score_sum = 0.0
    for index in tqdm(range(iteration)) :
        x_batch = x_test[batch_size * index : batch_size * (index+1)]
        y_batch_gt = y_test[batch_size * index : batch_size * (index +1)]
        y_pred = wrapper.inference(x_batch)
        
        top_k = 1
        num = y_pred.shape[-1]
        y_pred = y_pred.argsort()[:,-top_k:][:,::-1]
        y_pred = np.squeeze(y_pred)
        y_pred = np.expand_dims(np.eye(num)[y_pred], axis=0)
        
        score = sklearn.metrics.accuracy_score(y_batch_gt,y_pred)
        score_sum += score
    mean_score = score_sum / iteration
    result("TFLite model Accuracy: ", mean_score)


