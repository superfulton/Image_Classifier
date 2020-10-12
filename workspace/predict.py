import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import json
import numpy as np
import os


from PIL import Image

#Initailize the parser
parser = argparse.ArgumentParser(description='Flower Prediction App')

#Add Args
parser.add_argument('image_path',  action='store', help="Image Path")
    
parser.add_argument('saved_model', type=str, help="Trained Model")
    
parser.add_argument('--top_k', type=int, help='Probabilities of Top Classes')
    
parser.add_argument('--category_names', type=str, help='Names of Top Classes')
    
#Parse Arguments
args = parser.parse_args()

image_path = args.image_path
    
saved_model = args.saved_model
    
top_k = args.top_k
    
label_map = args.category_names


#Label Map
with open(label_map, 'r') as f:
      class_names = json.load(f)
          
          
def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = image.numpy()
    
    return image


def predict(image_path, model, top_k=5):
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    image = np.expand_dims(image, axis = 0)
        
    prediction = model.predict(image)
    probs, indexes = tf.math.top_k(prediction, k=top_k)
    indexes = indexes.numpy()
    
    probs = probs.numpy()[0]
    top_classes = [class_names[str(value+1)] for value in indexes[0]]
  
    
    return probs, top_classes
    
loaded_model = tf.keras.models.load_model(saved_model ,custom_objects={'KerasLayer':hub.KerasLayer})
loaded_model.summary()
      
probs, top_classes = predict(image_path, loaded_model, top_k=5)

    
print("Prediction for this {} using this {}" .format(image_path, saved_model))
print("------------------------------")
print("Here are the top predicted classes {}" .format(top_classes))
print("------------------------------")
print("Here are the top probabilities {} for the predicted classes {}" .format(probs, top_classes))
