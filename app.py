import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load your trained CNN model
model = tf.keras.models.load_model('C:/Users/Student/Desktop/dl ass/hema/model.h5')

st.title('Digit Recognition using CNN')

# Sidebar for user options
st.sidebar.title('Options')
option = st.sidebar.selectbox('Choose an option:', ['Draw Digit', 'Upload Image'])

if option == 'Draw Digit':
    st.sidebar.text('Draw a digit and click "Predict"')
    canvas = st.canvas(fill_color="#000000", stroke_width=10, stroke_color="#FFFFFF", background_color="#000000", height=150, width=150)
    predict_button = st.sidebar.button("Predict")

    if predict_button:
        # Convert the canvas drawing to an image
        digit_image = canvas.image_data.astype(np.uint8)

        # Resize the image to 28x28 pixels and preprocess it
        digit_image = cv2.resize(digit_image, (28, 28))
        digit_image = digit_image[:, :, 3]  # Extract alpha channel
        digit_image = digit_image / 255.0  # Normalize pixel values

        # Make a prediction
        digit_image = np.expand_dims(digit_image, axis=0)
        prediction = model.predict(digit_image)
        predicted_class = np.argmax(prediction)

        st.image(digit_image.squeeze(), caption='Drawn Digit', use_column_width=True)
        st.write(f'Predicted Digit: {predicted_class}')

elif option == 'Upload Image':
    st.sidebar.text('Upload an image and click "Predict"')
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    predict_button = st.sidebar.button("Predict")

    if uploaded_image is not None and predict_button:
        # Preprocess the uploaded image
        image = Image.open(uploaded_image)
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((28, 28))
        image = np.array(image)
        image = image / 255.0  # Normalize pixel values

        # Make a prediction
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction)

        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
        st.write(f'Predicted Digit: {predicted_class}')
