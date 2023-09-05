# app.py
import streamlit as st
import tensorflow as tf
import numpy as np

# Load your trained CNN model
model = tf.keras.models.load_model('model.h5')

st.title('Digit Recognition using CNN')

# Upload an image for prediction
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Preprocess the uploaded image
    image = tf.image.decode_image(uploaded_image.read(), channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (28, 28))
    image = np.expand_dims(image, axis=0)

    # Perform inference
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    st.write(f'Predicted Digit: {predicted_class}')

