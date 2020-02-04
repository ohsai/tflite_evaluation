from keras.preprocessing import image
import os
import numpy as np
from tex_print import *
from tqdm import tqdm


class ImageNetDataset :
  def __init__(self,preprocess_func, input_shape=(224,224)) :
# Parameters
    self.imagenet_root_dir = os.path.expanduser('~') + "/Space/imagenet/"
    self.imagenet_dir = self.imagenet_root_dir + "validation/noncategorized/"
    self.label_filepath = self.imagenet_root_dir +"devkit/caffe_ILSVRC2012_validation_ground_truth.txt"
    self.number_of_images = 1000
    self.input_shape = (input_shape[0],input_shape[1])
    self.number_of_classes = 1000
    self.preprocess_func = preprocess_func

  def load(self) :
    imagename_list = os.listdir(self.imagenet_dir)
    imagename_list.sort()
    imagepath_list = list(map(lambda x : self.imagenet_dir + x, imagename_list[:self.number_of_images]))
    image_list = [self.preprocess_image_from_path(f) for f in tqdm(imagepath_list)]
    debug("ImageNet Shape: ",
        image_list[0].shape,
        " | type: ",
        type(image_list[0]),
        " | # of images: " , 
        len(image_list))
    image_np = np.concatenate(image_list,axis=0) # list to numpy stacking

    label_list = []
    label_file = open(self.label_filepath)
    for line in label_file.readlines() :
        label_list.append(int(line.split()[1])) # label of i-th image is label_list[i]
    label_file.close()
    label_list = label_list[:self.number_of_images]
    label_np = np.eye(self.number_of_classes)[np.asarray(label_list)] # One-hot Encoding
    return image_np, label_np

  def preprocess_image_from_path(self,image_path) :
    image_t = image.load_img(image_path, target_size=self.input_shape)
    image_t = image.img_to_array(image_t)
    image_t = np.expand_dims(image_t, axis=0)
    image_t = self.preprocess_func(image_t)
    return image_t
  
 
