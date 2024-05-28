import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the models
@st.cache(allow_output_mutation=True)
def load_models():
    model_vgg16 = tf.keras.models.load_model('vgg16_model.h5')
    model_resnet50 = tf.keras.models.load_model('resnet50_model.h5')
    model_alexnet = tf.keras.models.load_model('alexnet_model.h5')
    return model_vgg16, model_resnet50, model_alexnet

model_vgg16, model_resnet50, model_alexnet = load_models()

# Function to load and preprocess image
def load_and_preprocess_image(img):
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Streamlit app
st.title('Image Classification with VGG16, ResNet50, and AlexNet')
st.write('Upload an image to classify using VGG16, ResNet50, and AlexNet models.')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    img_array = load_and_preprocess_image(img)

    # Get predictions from each model
    prediction_vgg16 = model_vgg16.predict(img_array)
    prediction_resnet50 = model_resnet50.predict(img_array)
    prediction_alexnet = model_alexnet.predict(img_array)

    # Get the class with the highest probability
    class_vgg16 = np.argmax(prediction_vgg16)
    class_resnet50 = np.argmax(prediction_resnet50)
    class_alexnet = np.argmax(prediction_alexnet)

    # Display the predictions
    st.write(f"VGG16 Prediction: Class {class_vgg16}, Confidence: {np.max(prediction_vgg16):.2f}")
    st.write(f"ResNet50 Prediction: Class {class_resnet50}, Confidence: {np.max(prediction_resnet50):.2f}")
    st.write(f"AlexNet Prediction: Class {class_alexnet}, Confidence: {np.max(prediction_alexnet):.2f}")

    # Plot the predictions
    fig, ax = plt.subplots()
    labels = ['VGG16', 'ResNet50', 'AlexNet']
    confidences = [np.max(prediction_vgg16), np.max(prediction_resnet50), np.max(prediction_alexnet)]
    ax.bar(labels, confidences)
    ax.set_ylabel('Confidence')
    ax.set_title('Model Predictions')
    st.pyplot(fig)
