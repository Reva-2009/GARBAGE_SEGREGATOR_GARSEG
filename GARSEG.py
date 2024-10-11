from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

# Load the model once (ensure "keras_model.h5" is in the correct directory)
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()
class_names = [name.strip() for name in class_names]  # Strip whitespace and newlines

def segregate_wastes(img):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Create the array of the right shape to feed into the Keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Resizing the image to be at least 224x224
    size = (224, 224)
    image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = round(prediction[0][index] * 100, 2)  # Convert to percentage with 2 decimal places

    return class_name, confidence_score

st.set_page_config(layout="wide")
st.title("GARBAGE SEGREGATOR - GARSEG")

input_img = st.file_uploader("ENTER YOUR IMAGE HERE!", type=['jpeg', 'jpg', 'png'])

if input_img is not None:
    if st.button("CLASSIFY"):
        # Load and process the uploaded image
        image_file = Image.open(input_img)

        # Display the uploaded image
        col1, col2 = st.columns([1, 1])
        with col1:
            st.info('YOUR UPLOADED IMAGE!')
            st.image(image_file, use_column_width=True)

        # Perform classification
        with col2:
            st.info('THE GARBAGE IS OF TYPE:')
            label, confidence_score = segregate_wastes(image_file)
            st.write(f"**Type of waste:** {label[2:]}")
            st.write(f"**Confidence Score:** {confidence_score}%")

    else:
        st.info('NO INPUT IMAGE RECEIVED! PLEASE UPLOAD A FILE FOR ANY RESULT.')
