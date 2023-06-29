from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

def waste_segregator(img):
    # Preprocess the image
    size = (224, 224)
    image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.expand_dims(normalized_image_array, axis=0)

    # Predict the waste type
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = round(prediction[0][index] * 100, 2)  # Convert to percentage with 2 decimal places

    return class_name, confidence_score

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

st.set_page_config(layout='wide')
st.title('GARBAGE SEGREGATOR-GARSEG')

input_img = st.file_uploader('ENTER YOUR IMAGE HERE!', type=['jpeg', 'jpg', 'png'])

if input_img is not None:
    if st.button('CLASSIFY'):
        # Perform classification or any desired action
        image_file = Image.open(input_img)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.info('YOUR UPLOADED IMAGE!')
            st.image(image_file, use_column_width=True)

        with col2:
            st.info('YOUR WASTE IS OF TYPE/RESULT')
            label, confidence_score = waste_segregator(image_file)
            st.write(label)
            st.write(confidence_score)
