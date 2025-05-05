import streamlit as st
import tensorflow as tf
import numpy as np
import time
import datetime
import random
import pandas as pd

st.set_page_config(
    page_title="Road Condition Detector",
    page_icon="🚣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model prediction function
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

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Road condition detection", "Gallery", "Project Report"])

st.sidebar.write("---")
st.sidebar.write("### Customization")
threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.3, 0.05)
show_advanced = st.sidebar.checkbox("Show Advanced Details", False)

# Theme
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

# Home
if app_mode == "Home":
    st.header("Road Condition Recognition with Python")
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown("""
Welcome to the **Road Condition Detection System**.
Use AI to detect road damage from photos, particularly useful in rainy conditions.
    """)

# About
elif app_mode == "About":
    st.header("About This Web App")
    st.markdown("""
This project uses a CNN to classify roads as **Good** or **Bad**.
Developed by:
- **K. M. Sirazul Monir** — ID: 200021247  
- **Shafin Ibnul Mohasin** — ID: 200021244  
- **Whiduzzaman Rishad** — ID: 200021228
    """)

# Detection
elif app_mode == "Road condition detection":
    st.header("Road Quality Detection")
    input_option = st.radio("Select input method:", ["Upload Image", "Use Camera"])
    test_image = st.file_uploader("Choose a road image:", type=["jpg", "jpeg", "png"]) if input_option == "Upload Image" else st.camera_input("Take a picture of the road")

    if 'history' not in st.session_state:
        st.session_state.history = []

    if test_image is not None:
        st.image(test_image, caption="Selected Image", use_column_width=True)
        if st.button("Analyze Road"):
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
                        st.success(f"Good Road with {confidence:.2%} confidence")
                        st.info("Appears safe for travel.")
                    else:
                        st.error(f"Bad Road with {confidence:.2%} confidence")
                        st.warning("May need maintenance.")
                    st.progress(int(confidence * 100))
                    if confidence > 0.8:
                        st.success(f"High confidence: {confidence:.2%}")
                    elif confidence > 0.5:
                        st.info(f"Medium confidence: {confidence:.2%}")
                    else:
                        st.warning(f"Low confidence: {confidence:.2%}")
                with tab2:
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
                    for item in st.session_state.history:
                        st.write(f"{item['timestamp']} - {item['result']} ({item['confidence']:.2%})")

                with st.expander("View on Map (Demo)"):
                    map_data = pd.DataFrame({
                        'lat': [random.uniform(40.0, 41.0)],
                        'lon': [random.uniform(-74.0, -73.0)]
                    })
                    st.map(map_data)
                    st.caption("Note: Random location")

# Gallery
elif app_mode == "Gallery":
    st.header("🖼️ Road Condition Gallery")
    st.markdown("Browse categorized road images used for model training:")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✅ Good Roads")
        st.image("good1.jpg", caption="Smooth asphalt", use_column_width=True)
        st.image("good2.jpg", caption="Clean road surface", use_column_width=True)
        st.image("good3.jpg", caption="Well-maintained road", use_column_width=True)

    with col2:
        st.subheader("⚠️ Bad Roads")
        st.image("bad1.jpg", caption="Potholes", use_column_width=True)
        st.image("bad2.jpg", caption="Waterlogging", use_column_width=True)
        st.image("bad3.jpg", caption="Cracked surface", use_column_width=True)

    st.info("Try your own images in 'Road condition detection' tab.")

# Project Report
elif app_mode == "Project Report":
    st.header("📄 Project Report")
    st.markdown("""
### Methodology
- Data collected from real road conditions
- Images augmented for weather and lighting
- Trained using CNN model with binary classification (Good/Bad)

### Experimental Setup
- **Framework**: TensorFlow + Keras
- **Hardware**: Trained on GPU
- **Image size**: 256x256 pixels

### Results
- Accuracy: ~92% on test data
- Works best under clear or moderately rainy conditions

### Challenges
- Differentiating wet patches from potholes
- Noise in low-light images

### Future Work
- Mobile app integration
- Crowdsourced reporting system
- GIS-based road tracking and logging

### References
- Public pothole datasets
- TensorFlow and Streamlit documentation
- Research papers on road condition monitoring
    """)
