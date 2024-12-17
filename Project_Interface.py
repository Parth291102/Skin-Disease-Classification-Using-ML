import gradio as gr
import os
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow as tf

# Load the model and define class names
model = tf.keras.models.load_model('/content/drive/MyDrive/resnet50FinedTunedCNN.keras')
class_names = ['Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Monkeypox']

# Define the image processing and prediction function
def image_mod(image_path):
    # Load the image from the file path
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))  # Match model input size
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the class of the image
    pred = model.predict(img_array)
    class_id = np.argmax(pred, axis=1)

    # Get the class name
    predicted_class = class_names[class_id[0]]
    return predicted_class

# Create the Gradio interface
demo = gr.Interface(
    fn=image_mod,
    inputs=gr.Image(type='filepath'),
    outputs=gr.Textbox(label="Predicted Class"),
    examples=[
        "/content/drive/MyDrive/Original Images/FOLDS/fold5/Test/Healthy/HEALTHY_104_01.jpg",
        "/content/drive/MyDrive/Original Images/FOLDS/fold5/Test/HFMD/HFMD_118_01.jpg",
        "/content/drive/MyDrive/Original Images/FOLDS/fold5/Test/Chickenpox/CHP_29_01.jpg",
        "/content/drive/MyDrive/Original Images/FOLDS/fold5/Test/Cowpox/CWP_22_02.jpg",
        "/content/drive/MyDrive/Original Images/FOLDS/fold5/Test/Measles/MSL_27_01.jpg",
        "/content/drive/MyDrive/Original Images/FOLDS/fold5/Test/Monkeypox/MKP_141_01.jpg",
    ],
)

if __name__ == "__main__":
    demo.launch()
