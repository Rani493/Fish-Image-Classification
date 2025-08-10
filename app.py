# ==============================================================================
# --- Streamlit Application (app.py) -------------------------------------------
# ==============================================================================
# This script creates a web application to use a trained deep learning model
# for image classification.
#
# To run this app, make sure you have Streamlit installed:
# pip install streamlit
#
# Then, navigate to your project directory in the terminal and run:
# streamlit run app.py
#

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# --- Streamlit App Configuration ---
# `st.set_page_config()` must be the very first Streamlit command.
st.set_page_config(page_title="Fish Classifier", layout="centered")

# --- Configuration ---
# You must run the training script first to create these files.
MODEL_PATH = 'fish_classifier_transfer.h5'  # Change this to 'fish_classifier_cnn.h5' to use your custom CNN model
IMG_SIZE = (224, 224)

# IMPORTANT: Update this list with your actual class names in the correct order.
# The order must match the numerical indices assigned by the data generator.
class_names = ['Category_1_Name', 'Category_2_Name', 'Category_3_Name', 'Category_4_Name', 'Category_5_Name',
               'Category_6_Name', 'Category_7_Name', 'Category_8_Name', 'Category_9_Name', 'Category_10_Name',
               'Category_11_Name']


# Load the trained model from the saved file.
# The @st.cache_resource decorator ensures the model is loaded only once.
@st.cache_resource
def load_my_model(model_path):
    """Loads a pre-trained Keras model from a file."""
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None


# Load the model and display an error if it's not found
model = load_my_model(MODEL_PATH)
if model is None:
    st.stop()

# --- Streamlit App Layout ---
st.title('🐟 Fish Image Classifier')
st.markdown('Upload an image of a fish to get its classification!')

# File uploader widget
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("---")
        st.write("### Classifying...")

        # Preprocess the image for the model
        image = image.resize(IMG_SIZE)
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)  # Create a batch
        img_array = img_array / 255.0  # Rescale pixel values

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_index]

        predicted_class = class_names[predicted_class_index]

        st.success(f'Prediction: **{predicted_class}**')
        st.write(f'Confidence: **{confidence:.2f}**')

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
