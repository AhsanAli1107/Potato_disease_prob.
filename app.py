import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the model
def load_model_from_file():
    model = load_model('/Users/ahsanali/Desktop/Potato_disease_prob./potato_disease.keras')
    return model

model = load_model_from_file()

# Streamlit app
st.title('Potato Disease Classification')
file = st.file_uploader('Please upload an image', type=['png', 'jpg'])

# Prediction function
def import_and_predict(image_data, model):
    size = (256, 256)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image) / 255.0  # Normalize
    img_reshape = img[np.newaxis, ...]  # Add batch dimension
    predictions = model.predict(img_reshape)
    return predictions

if file is not None:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = ['Early_blight', 'Healthy', 'Late_blight']
    result = class_names[np.argmax(predictions)]
    st.success(f"This image most likely is: {result}")
else:
    st.write("Please upload an image.")

