import streamlit as st
import tensorflow as tf
import numpy as np
import time
import datetime
import random
import pandas as pd

st.set_page_config(
    page_title="Road Condition Detector",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# model prediction function
def model_prediction(test_image):
    model = tf.keras.models.load_model("road_pothole_Rainy_days.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(256, 256))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    raw_prediction = model.predict(input_arr)[0][0]
    threshold = 0.3
    predicted_class = 1 if raw_prediction > threshold else 0
    class_names = ["BAD ROAD", "Good road"]
    result = class_names[predicted_class]
    confidence = raw_prediction if predicted_class == 1 else (1 - raw_prediction)
    return result, confidence

# Sidebar ------------------------------------------------------------------------------------------------------
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Gallery", "Project Report", "Road condition detection"])

st.sidebar.write("---")
st.sidebar.write("### Customization")
threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.3, 0.05)
show_advanced = st.sidebar.checkbox("Show Advanced Details", False)

# Main Page
if app_mode == "Home":
    st.header("Road Condition Recognition with Python")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
Welcome to the **Road Condition Detection System** — a simple, smart, and efficient web application that helps detect whether a road surface is in **good** or **bad** condition using an AI-powered image classifier.
... (rest of Home markdown remains unchanged)
    """)
    theme = st.sidebar.selectbox("Choose Theme", ["Light", "Dark", "Blue"])
    if theme == "Dark":
        st.markdown("""
            <style>
            .stApp {
                background-color: #121212;
                color: white;
            }
            </style>
        """, unsafe_allow_html=True)
    elif theme == "Blue":
        st.markdown("""
            <style>
            .stApp {
                background-color: #0e1117;
                color: #f1f1f1;
            }
            </style>
        """, unsafe_allow_html=True)

elif app_mode == "About":
    st.header(" About This Web App")
    st.markdown("""
Welcome to the **Road Condition Detector** — an AI-powered web application designed to assess the quality of road surfaces from images.
... (rest of About markdown remains unchanged)
    """, unsafe_allow_html=True)

elif app_mode == "Gallery":
    st.header("Project Gallery")
    st.markdown("This section will display a gallery of example road condition images.")
    st.image(["sample1.jpg", "sample2.jpg", "sample3.jpg"], width=300)

elif app_mode == "Project Report":
    st.header("Project Report")
    st.markdown("This is the section where you can attach or write the full project report.")

elif app_mode == "Road condition detection":
    st.header("Road Quality Detection")
    input_option = st.radio("Select input method:", ["Upload Image", "Use Camera"])
    if input_option == "Upload Image":
        test_image = st.file_uploader("Choose a road image:", type=["jpg", "jpeg", "png"])
    else:
        test_image = st.camera_input("Take a picture of the road")
    if test_image is not None:
        st.image(test_image, caption="Selected Image", use_column_width=True)
    if 'history' not in st.session_state:
        st.session_state.history = []
    if test_image is not None and st.button("Analyze Road"):
        with st.spinner("Analyzing image..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            result, confidence = model_prediction(test_image)
            progress_bar.empty()
            st.balloons()
            tab1, tab2 = st.tabs(["Results", "Technical Details"])
            with tab1:
                if result == "Good road":
                    st.success(f"The model predicts this is a GOOD ROAD with {confidence:.2%} confidence")
                    st.info("This road appears to be in good condition with no chances of clogged water formation in rainy seasons")
                else:
                    st.error(f"The model predicts this is a damaged ROAD with {confidence:.2%} confidence")
                    st.warning("This road has a high probabily of clogged water formation in rainy seasons thus may need maintenance.")
                st.write("Confidence Level:")
                st.progress(int(confidence * 100))
                if confidence > 0.8:
                    st.success(f"High confidence: {confidence:.2%}")
                elif confidence > 0.5:
                    st.info(f"Medium confidence: {confidence:.2%}")
                else:
                    st.warning(f"Low confidence: {confidence:.2%}")
            with tab2:
                st.write("#### Technical Details")
                st.write(f"- Classification: {result}")
                st.write(f"- Confidence Score: {confidence:.2%}")
                st.write("- Model: TensorFlow CNN")
                st.write("- Threshold used: 0.3")
            st.session_state.history.append({
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "result": result,
                "confidence": confidence
            })
            with st.expander("Analysis History"):
                for i, item in enumerate(st.session_state.history):
                    st.write(f"{item['timestamp']} - {item['result']} ({item['confidence']:.2%})")
            with st.expander("View on Map (Demo)"):
                map_data = pd.DataFrame({
                    'lat': [random.uniform(40.0, 41.0)],
                    'lon': [random.uniform(-74.0, -73.0)]
                })
                st.map(map_data)
                st.caption("Note: This is a demo map with random coordinates")
