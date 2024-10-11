from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

# Initialize the model as None
model = None

def load_waste_model():
    global model
    if model is None:
        try:
            # Load the model only if it is not loaded
            model = load_model("keras_model.h5", compile=False)
            print("Model loaded successfully.")  # Debugging line
        except Exception as e:
            print(f"Error loading model: {e}")  # Print the error for debugging
            st.error(f"Failed to load the model: {e}")  # Display error in Streamlit

def waste_segregator(img):
    # Load the model if it's not loaded (handles inactivity issue)
    load_waste_model()
    
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

# Load the labels
try:
    class_names = open("labels.txt", "r").readlines()
    class_names = [name.strip() for name in class_names]  # Remove any whitespace
except Exception as e:
    print(f"Error loading labels: {e}")
    st.error(f"Failed to load labels: {e}")

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
