import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress most TensorFlow warnings

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image  # Ensure proper image handling

# Function to load the model
def load_model():
    try:
        model = tf.keras.models.load_model("trained_model.h5")
        print("Loaded trained_model.h5")
    except Exception as e:
        print(f"Error loading trained_model.h5: {e}")
        st.error("Could not load the model. Ensure 'trained_model.h5' is in the correct folder.")
        return None
    return model


# Model Prediction Function
def model_prediction(model, test_image):
    if model is None:
        return None

    # Open and preprocess the image
    image = Image.open(test_image).convert("RGB")
    image = image.resize((128, 128))
    input_arr = np.array(image) / 255.0  # Normalize
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(input_arr)

    # Print raw prediction probabilities
    print("Prediction Probabilities:", prediction)

    result_index = np.argmax(prediction)  # Get highest probability class
    print("Predicted Index:", result_index)

    return result_index

# Load the model at the start
model = load_model()

# Streamlit Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases.

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant.
    2. **Analysis:** Our system processes the image using advanced AI models.
    3. **Results:** View the results and recommendations.

    ### Features:
    - **High Accuracy** 🎯
    - **User-Friendly Interface** ✅
    - **Fast & Efficient Analysis** 🚀

    ### Get Started:
    Click on the **Disease Recognition** page in the sidebar to begin!
    """)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### Dataset Information:
    - **87K RGB images** of healthy and diseased crop leaves.
    - **38 different plant disease categories**.
    - **Dataset Split:** 80% Training, 20% Validation.

    #### Purpose:
    This project aims to assist farmers in identifying crop diseases early, helping them take preventive action before significant damage occurs.
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])
    
    if test_image:
        st.image(test_image, use_column_width=True, caption="Uploaded Image")

        if st.button("Predict"):
            with st.spinner("Analyzing..."):
                result_index = model_prediction(model, test_image)
                if result_index is not None:
                    # Define Class Labels
                    class_name = [
                        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                        'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                        'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                        'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                        'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                        'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
                    ]
                    st.success(f"Model Prediction: **{class_name[result_index]}** 🌱✅")
                else:
                    st.error("Prediction failed. Check model and input image.")

