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



#model prediction function
def model_prediction(test_image):
    model = tf.keras.models.load_model("road_pothole_Rainy_days.keras")
    
    # Load and preprocess the image
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(256, 256))  # Adjust size to match your model
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    
    input_arr = np.array([input_arr])  # Convert single image to batch
    
    # Get prediction (raw probability)
    raw_prediction = model.predict(input_arr)[0][0]  # Single value between 0 and 1
    
    # Apply the same threshold you used during validation 
    threshold=0.3
    predicted_class = 1 if raw_prediction > threshold else 0
    
    # Map to class names
    class_names = ["BAD ROAD", "Good road"]
    result = class_names[predicted_class]
    
    # For confidence, use the raw prediction or its complement depending on the class
    confidence = raw_prediction if predicted_class == 1 else (1 - raw_prediction)
    
    return result, confidence

#Sidebar------------------------------------------------------------------------------------------------------

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Road condition detection"])

# Add customization options
st.sidebar.write("---")
st.sidebar.write("### Customization")
threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.3, 0.05)
show_advanced = st.sidebar.checkbox("Show Advanced Details", False)
#Main Page
if(app_mode=="Home"):
    st.header("Road Condition Recognition with Python")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)

    st.markdown("""
Welcome to the **Road Condition Detection System** — a simple, smart, and efficient web application that helps detect whether a road surface is in **good** or **bad** condition using an AI-powered image classifier.

Our goal is to support road safety and maintenance efforts by enabling quick on-the-spot assessments using nothing more than a photo — either captured directly or uploaded.

---
###  What This App Can Do
- Detect potholes and road damages in real-time
- Show prediction confidence with visual feedback
- Use your camera or upload road images
- Adjust detection sensitivity to suit different environments
- Visualize demo results on a map
- Customize the app’s appearance with themes

---
### Under the Hood
This tool uses a **Convolutional Neural Network (CNN)** trained with augmented images to handle real-world conditions, especially in **rainy environments** where potholes become a major safety concern.

---
###  How to Use
1. Go to the **"Road Condition Detection"** tab from the sidebar.
2. Upload a road image or take a photo.
3. Click **"Analyze Road"** to get instant results.
4. Check the results, confidence, and view previous analyses in the history log.

---
### Customization
Use the sidebar to:
- Set your **confidence threshold**
- Change **themes**
- Toggle **advanced details**

---
Start exploring, analyzing, and making your roads safer!
    """)

# Add theme selector
theme = st.sidebar.selectbox(
    "Choose Theme",
    ["Light", "Dark", "Blue"]
)

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
#About Project
elif(app_mode=="About"):
    st.header(" About This Web App")
    st.markdown("""
Welcome to the **Road Condition Detector** — an AI-powered web application designed to assess the quality of road surfaces from images. Whether you're capturing a photo on-site or uploading an image, our system provides fast and reliable feedback on the road’s condition.

This tool is particularly useful for detecting **damaged or pothole-ridden roads**, especially in **rainy weather scenarios** where early maintenance can prevent water clogging and accidents.

---

### How It Works

The app uses a **deep learning model (CNN)** trained on a curated dataset of road images. It classifies the input image into two categories:

-  **Good Road** – Safe, smooth, and well-maintained.  
-  **Bad Road** – Likely damaged, cracked, or susceptible to water accumulation.

A customizable **threshold slider** lets you adjust the model's sensitivity, and each prediction includes a **confidence score** for transparency.

---

###  Key Features

-  Upload or capture images directly  
-  Confidence-based classification  
-  Adjustable detection threshold  
-  Real-time prediction history  
-  Demo map integration  
-  Theme customization (Light, Dark, Blue)

---

###  About the Dataset

The model is trained on a dataset recreated using offline augmentation techniques. The original dataset is publicly available on GitHub and focuses on pothole detection under various environmental conditions.

---

###  Future Possibilities

This system can be integrated into:
- Smart city monitoring tools  
- Municipality road reporting systems  
- Road maintenance scheduling dashboards  
- Crowdsourced infrastructure reporting apps

---

### Developed By

<div style='border: 1px solid #ccc; padding: 10px; border-radius: 10px; background-color: #002a47'>

**K. M. Sirazul Monir** — ID: 200021247  📷  
**Shafin Ibnul Mohasin** — ID: 200021244  🧠  
**Whiduzzaman Rishad** — ID: 200021228  🔧  

> Built as part of an academic project at Islamic University of Technology. 
> All team members contributed collaboratively across development, research, and system design.

</div>

---

Explore the app and see it in action!
    """, unsafe_allow_html=True)


# Prediction Page
elif(app_mode == "Road condition detection"):
    st.header("Road Quality Detection")
    
    # Let the user choose between uploading an image or using the camera
    input_option = st.radio("Select input method:", ["Upload Image", "Use Camera"])
    
    if input_option == "Upload Image":
        test_image = st.file_uploader("Choose a road image:", type=["jpg", "jpeg", "png"])
    else:
        test_image = st.camera_input("Take a picture of the road")
    
    # Show the image when available
    if test_image is not None:
        st.image(test_image, caption="Selected Image", use_column_width=True)
    
    # Add this near the top of your script (for history feature)
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Predict button
    if test_image is not None and st.button("Analyze Road"):
        import time  # Add this for the progress bar
        
        with st.spinner("Analyzing image..."):
            # Show a progress bar for visual feedback
            progress_bar = st.progress(0)
            for i in range(100):
                # Simulate progress while your model is actually working
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Get prediction using your model_prediction function
            result, confidence = model_prediction(test_image)
            progress_bar.empty()  # Remove progress bar when done
            
            # Add some visual effects for user engagement
            st.balloons()
            
            # Display the result with tabs
            tab1, tab2 = st.tabs(["Results", "Technical Details"])
            
            with tab1:
                # Show different messages based on the result
                if result == "Good road":
                    st.success(f"The model predicts this is a GOOD ROAD with {confidence:.2%} confidence")
                    st.info("This road appears to be in good condition with no chances of clogged water formation in rainy seasons")
                    # Uncomment for audio feedback
                    # st.audio("https://www.soundjay.com/buttons/sounds/button-09.mp3", format="audio/mp3")
                else:
                    st.error(f"The model predicts this is a damaged ROAD with {confidence:.2%} confidence")
                    st.warning("This road has a high probabily of clogged water formation in rainy seasons thus may need maintenance.")
                    # Uncomment for audio feedback
                    # st.audio("https://www.soundjay.com/buttons/sounds/button-10.mp3", format="audio/mp3")
                
                # Visual confidence meter
                st.write("Confidence Level:")
                st.progress(int(confidence * 100))
                
                # Color coding based on confidence
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
            
            # Save to history
            import datetime
            st.session_state.history.append({
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "result": result,
                "confidence": confidence
            })
            
            # Show history in expander
            with st.expander("Analysis History"):
                for i, item in enumerate(st.session_state.history):
                    st.write(f"{item['timestamp']} - {item['result']} ({item['confidence']:.2%})")
            
            # Add a simple map component
            with st.expander("View on Map (Demo)"):
                import random
                import pandas as pd
                # This is just a placeholder - in a real app you would use GPS coordinates
                map_data = pd.DataFrame({
                    'lat': [random.uniform(40.0, 41.0)],
                    'lon': [random.uniform(-74.0, -73.0)]
                })
                st.map(map_data)
                st.caption("Note: This is a demo map with random coordinates")
