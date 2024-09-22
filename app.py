import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('models/mnist_model.h5')

def preprocess_image(image):
    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))
    # Convert to grayscale
    image = image.convert('L')
    # Convert to numpy array and normalize
    image = np.array(image).astype('float32') / 255
    # Reshape to (1, 28, 28, 1) for model input
    image = image.reshape(1, 28, 28, 1)
    return image

def predict_digit(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    # Make prediction
    prediction = model.predict(processed_image)
    # Get the predicted digit
    digit = np.argmax(prediction)
    return digit

st.title("MNIST Digit Recognizer")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Predict'):
            digit = predict_digit(image)
            st.write(f"The predicted digit is: {digit}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

st.write("Note: Upload an image of a handwritten digit (0-9) for prediction.")
