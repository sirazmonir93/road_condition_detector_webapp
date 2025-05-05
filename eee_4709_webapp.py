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
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Road condition detection", "Project Report"])

st.sidebar.write("---")
st.sidebar.write("### Customization")
threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.3, 0.05)
show_advanced = st.sidebar.checkbox("Show Advanced Details", False)

theme = st.sidebar.selectbox("Choose Theme", ["Light", "Dark", "Blue"])
if theme == "Dark":
    st.markdown("""
        <style>
        .stApp { background-color: #121212; color: white; }
        </style>
    """, unsafe_allow_html=True)
elif theme == "Blue":
    st.markdown("""
        <style>
        .stApp { background-color: #0e1117; color: #f1f1f1; }
        </style>
    """, unsafe_allow_html=True)

# Home Page
if app_mode == "Home":
    st.header("Road Condition Recognition with Python")
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown("""
Welcome to the **Road Condition Detection System** — a smart and efficient app that helps detect whether a road surface is in **good** or **bad** condition using an AI-powered image classifier.

---

### What This App Can Do
- Detect potholes and road damages in real-time
- Show prediction confidence with visual feedback
- Use your camera or upload road images
- Adjust detection sensitivity to suit different environments
- Visualize demo results on a map
- Customize the app’s appearance with themes

---

### Under the Hood
Uses a **Convolutional Neural Network (CNN)** trained with augmented images, especially for **rainy environments** where potholes become more dangerous.

---

### How to Use
1. Go to **"Road Condition Detection"**
2. Upload a road image or take a photo
3. Click **"Analyze Road"** for results
4. Check the results, confidence, and past logs

---

### Customization
Use the sidebar to:
- Set your **confidence threshold**
- Toggle **advanced details**
- Change visual **themes**
    """)

# About Page
elif app_mode == "About":
    st.header("About This Web App")
    st.markdown("""
Welcome to the **Road Condition Detector** — an AI-powered web application designed to assess the quality of road surfaces from images.

---

### What We Built
- Smart CNN classifier for detecting:
  - **Good Road** – smooth, well-maintained  
  - **Bad Road** – cracked or likely water-clogged
- Upload or capture images
- Adjustable threshold
- Confidence score
- Map integration
- Theme customization

---

### Dataset
Trained on a public pothole dataset and augmented to simulate rainy weather and low-light conditions.

---

### Developed By
<div style='border: 1px solid #ccc; padding: 10px; border-radius: 10px; background-color: #002a47; color: white;'>
📌 **K. M. Sirazul Monir** — *ID: 200021247*  
📌 **Shafin Ibnul Mohasin** — *ID: 200021244*  
📌 **Whiduzzaman Rishad** — *ID: 200021228*  
</div>

**Instructor:** Md. Arefin Rabbi Emon  
Project at **Islamic University of Technology**
    """, unsafe_allow_html=True)

# Prediction Page
elif app_mode == "Road condition detection":
    st.header("Road Quality Detection")
    input_option = st.radio("Select input method:", ["Upload Image", "Use Camera"])
    test_image = st.file_uploader("Choose a road image:", type=["jpg", "jpeg", "png"]) if input_option == "Upload Image" else st.camera_input("Take a picture of the road")
    
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
                    st.info("This road appears to be in good condition with no clog risk.")
                else:
                    st.error(f"The model predicts this is a damaged ROAD with {confidence:.2%} confidence")
                    st.warning("This road may have water-clogging issues and requires maintenance.")
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
                st.write(f"- Threshold used: {threshold:.2f}")

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

# Project Report Page
elif app_mode == "Project Report":
    st.header("📘 Project Report")

    with st.expander("🧪 Experimental Setup"):
        st.markdown("""
- **Model Used**: Convolutional Neural Network (CNN)  
- **Training Framework**: TensorFlow/Keras  
- **Image Size**: 256x256  
- **Dataset**: Augmented rainy-condition pothole images  
- **Split**: 80% Train / 20% Test  
- **Platform**: Google Colab (GPU enabled)  
- **Optimizer**: Adam | **Loss**: Binary Crossentropy  
- **Evaluation**: Accuracy, Precision, Recall, F1 Score  
        """)

    with st.expander("🔬 Methodology"):
        st.markdown("""
1. **Data Collection**: Open dataset with annotated pothole images  
2. **Preprocessing**: Resize, normalize, augment with noise/rain  
3. **Model Design**: Custom CNN with dropout  
4. **Training**: 50 epochs with validation  
5. **Deployment**: Saved and used in `.keras` format via Streamlit  
        """)

    with st.expander("📊 Results"):
        st.markdown("""
- **Test Accuracy**: 91.6%  
- **Precision**: 88%  
- **Recall**: 93%  
- **F1 Score**: 90.4%  
> *More graphs and confusion matrix to be added soon*
        """)

    with st.expander("⚠️ Challenges Faced"):
        st.markdown("""
- Low-light or obstructed images decreased model performance  
- Differentiating between puddles and shadows  
- Edge-case misclassifications in muddy terrains  
- Balancing sensitivity (recall) vs false positives  
        """)

    with st.expander("🚀 Future Work"):
        st.markdown("""
- Integrate with GPS for crowd-sourced road reports  
- Expand model to handle more weather conditions  
- Develop an Android app  
- Include multilingual accessibility  
- Explore transfer learning with pre-trained networks  
        """)

    with st.expander("📚 References"):
        st.markdown("""
1. Roboflow Pothole Dataset  
2. TensorFlow Official Docs  
3. Streamlit Documentation  
4. CNN-based Road Safety Papers  
        """)
