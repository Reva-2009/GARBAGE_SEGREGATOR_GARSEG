from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

# Initialize the model as None
model = None

def load_waste_model():
    global model
    if model is None:
        # Load the model only if it is not loaded
        model = load_model("keras_model.h5", compile=False)
        print("Model loaded successfully.")  # Debugging line

def waste_segregator(img):
    # Load the model if it's not loaded (handles inactivity issue)
    load_waste_model()
    
    # Preprocess the image
    size = (224, 224)
    image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.expand_dims(normalized_image_array, axis=0)

    print("Image preprocessed successfully.")  # Debugging line
    try:
        # Predict the waste type
        prediction = model.predict(data)
        print(f"Prediction Raw Output: {prediction}")  # Debugging line
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = round(prediction[0][index] * 100, 2)  # Convert to percentage with 2 decimal places

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None  # Return None values in case of an error

    return class_name, confidence_score

# Load the labels
with open("labels.txt", "r") as file:
    class_names = file.read().splitlines()  # Load class names into a list

st.set_page_config(layout='wide')
st.title('GARBAGE SEGREGATOR - GARSEG')

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
            if label and confidence_score:  # Check if the label and confidence_score are valid
                st.write(label)
                st.write(f"Confidence Score: {confidence_score}%")
            else:
                st.error("Failed to classify the image.")

# Run this code and check the terminal output for any issues or results.
