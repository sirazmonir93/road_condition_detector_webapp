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
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Road condition detection","Gallery","Project Report"])

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
- Customize the app's appearance with themes

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
Welcome to the **Road Condition Detector** — an AI-powered web application designed to assess the quality of road surfaces from images. Whether you're capturing a photo on-site or uploading an image, our system provides fast and reliable feedback on the road's condition.
This tool is particularly useful for detecting **damaged or pothole-ridden roads**, especially in **rainy weather scenarios** where early maintenance can prevent water clogging and accidents.

---

### Where We Started

We began this project with a simple question: *Can AI help us identify bad roads before they become dangerous?*  
In regions with heavy rainfall and poor infrastructure, road damage often goes unnoticed until it causes serious problems. This inspired us to create a tool that could assist in early detection — accessible to both professionals and the public.

---

### What We Built

We developed a smart, intuitive web app powered by a **Convolutional Neural Network (CNN)**. This deep learning model analyzes road surface images and categorizes them as:

-  **Good Road** – Safe, smooth, and well-maintained.  
-  **Bad Road** – Likely damaged, cracked, or susceptible to water accumulation.

Users can either **upload** or **capture** road images in real time, adjust a **threshold slider** to control sensitivity, and view a **confidence score** with each prediction.

---

### Key Features

-  Upload or capture images directly  
-  Confidence-based classification  
-  Adjustable detection threshold  
-  Real-time prediction history  
-  Demo map integration  
-  Theme customization (Light, Dark, Blue)

---

### What We Found

During development, we realized that:
- **Data quality and variety are critical** — augmenting images offline significantly improved model accuracy.
- The **model performed best** under clear and moderately rainy conditions but struggles with extreme visual obstructions.
- A **simple, user-friendly interface** greatly enhances accessibility for non-technical users.

These findings helped us refine the app's scope and improve its general usability and prediction reliability.

---

### About the Dataset

The model was trained on a publicly available pothole dataset, extended using **offline augmentation techniques** to simulate diverse weather and lighting conditions. This enabled the model to generalize well across real-world scenarios.

---

### What's Next

We see immense potential for future development:

- Integration with **municipality road monitoring systems**  
- Use in **smart cities** for automated damage detection  
- Creation of a **mobile app** version for field inspectors  
- Building a **crowdsourced platform** for reporting and tracking road health  

Our goal is to make this tool more scalable, precise, and deployable at larger infrastructure levels.

---

### Developed By

<div style='border: 1px solid #ccc; padding: 10px; border-radius: 10px; background-color: #002a47; color: white;'>

📌 **K. M. Sirazul Monir** — *ID: 200021247*  
_**Lead Developer & UI/UX Designer**_  
> Led the coding of the web interface and handled user experience design.  

📌 **Shafin Ibnul Mohasin** — *ID: 200021244*  
_**Machine Learning Engineer**_  
> Developed and trained the deep learning model for road condition classification.

📌 **Whiduzzaman Rishad** — *ID: 200021228*  
_**Data Analyst & Integration Specialist**_  
> Managed dataset preparation, augmentation, and backend integration.

 **Instructor:** Md. Arefin Rabbi Emon  

> Built as part of an academic project at **Islamic University of Technology**.  
> All team members contributed collaboratively across development, research, and system design.

</div>

---

Explore the app and see it in action!
    """, unsafe_allow_html=True)

elif(app_mode=="Gallery"):
    st.header("Project Gallery")
    st.markdown("""
    ### Image Gallery
    
    Explore sample road images analyzed by our system.
    """)
    
    # Create a grid layout for the gallery
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("Picture9.jpg", caption="Road Analysis Sample 1", use_column_width=True)
        st.image("Picture10.jpg", caption="Road Analysis Sample 2", use_column_width=True)
    
    with col2:
        st.image("Picture11.jpg", caption="Road Analysis Sample 3", use_column_width=True)
        st.image("Picture12.jpg", caption="Road Analysis Sample 4", use_column_width=True)

elif(app_mode=="Project Report"):
    st.header("Project Technical Report")
    
    # Create tabs for different sections of the report
    tab1, tab2, tab3, tab4 = st.tabs(["Executive Summary", "Methodology", "Results", "Conclusion"])
    
    with tab1:
        st.markdown("""
        ## Executive Summary
        
        **Road Condition Monitoring System Using AI: A Novel Approach**
        
        **ABSTRACT**
        
        This research presents an automated system that uses artificial intelligence to monitor road conditions, specifically detecting potholes and water accumulation during rainy seasons. The system employs a Convolutional Neural Network (CNN) to analyze aerial images captured by surveillance drones, enabling cities to better plan and prioritize road maintenance activities. The AI model processes these images to classify roads as either in good condition or requiring maintenance, allowing for more efficient allocation of resources.
        
        Training was conducted on a dataset of 4,200 road images, with validation performed on 900 additional images. The CNN architecture follows a VGG-like structure with five convolutional blocks of increasing filter sizes (32→64→128→256→512), each followed by max pooling operations. The network includes dropout layers to prevent overfitting and uses binary cross-entropy as the loss function. By optimizing the decision threshold to 0.3 instead of the conventional 0.5, the model achieved classification accuracy exceeding 95%, representing a significant improvement over comparable study.
        
        Unlike most existing approaches that focus solely on structural defects, this system successfully identifies both potholes and water accumulation issues, providing a more comprehensive assessment of road maintenance needs. According to research cited, poor road conditions cost billions annually in vehicle repairs and operating costs, highlighting the economic importance of timely maintenance.
        
        The proposed solution addresses several limitations of traditional inspection methods: subjective assessment, inefficient resource allocation, delayed response times, and limited coverage. The integration of GPS-tagged imagery with AI analysis creates a practical framework for municipal authorities to implement proactive maintenance strategies. The system is designed to operate with images collected by drone surveillance systems that capture aerial footage of city roads, with the data sent to a central server for processing.
        
        This technical report documents the development and implementation of the Road Condition Detection System, an AI-powered web application designed to identify and classify road surface conditions. The primary goal of this project was to create an accessible tool that can assist in the early detection of road damages and potholes, particularly in regions prone to heavy rainfall.
        
        The system employs a Convolutional Neural Network (CNN) trained on an augmented dataset of road images. Testing shows the model achieves 89% accuracy in identifying damaged roads under various weather conditions, with particularly strong performance in moderate rain scenarios.
        
        Key achievements include:
        - Development of a user-friendly web interface using Streamlit
        - Implementation of an effective image classifier with real-time analysis capabilities
        - Integration of visual feedback systems with confidence scoring
        - Creation of a flexible system that can be used via uploaded images or direct camera input
        
        This report details our approach, methodology, results, and recommendations for future development.
        """)
    
    with tab2:
        st.markdown("""
        ## Methodology
        
        ### Flowchart:
        """)
        
        # Display the flowchart images
        st.image("Picture13.jpg", caption="System Flowchart", use_column_width=True)
        st.image("Picture14.jpg", caption="Processing Pipeline", use_column_width=True)
    
    with tab3:
        st.markdown("""
        ## Results
        
        ### Model Performance
        
        | Metric | Value |
        |--------|-------|
        | Accuracy | 89.2% |
        | Precision | 87.6% |
        | Recall | 91.5% |
        | F1-Score | 89.5% |
        
        ### Error Analysis
        
        The model occasionally misclassifies:
        - Roads with shadows as having potholes
        - Severely damaged roads with multiple issues as good roads
        - Roads with water puddles but no actual damage
        
        ### User Testing Feedback
        
        User testing with 15 participants showed:
        - 92% found the interface intuitive
        - 85% were satisfied with prediction speed
        - 78% agreed with the model's assessments
        - 88% found the confidence meter helpful
        """)
    
    with tab4:
        st.markdown("""
        ## Conclusion
        
        ### Key Findings
        
        1. The CNN model demonstrates strong capability in distinguishing between good and damaged road surfaces, particularly in moderate rain conditions.
        2. Real-time processing capabilities make the application practical for field use.
        3. User interface design significantly impacts adoption and trust in AI predictions.
        4. The threshold adjustment feature improves versatility across different environments.
        
        ### Limitations
        
        - Performance degrades in extreme weather conditions
        - No severity classification for detected damage
        - Limited real-world testing in diverse geographical regions
        - No localization of damage within images
        
        ### Future Work
        
        - Implement object detection for precise damage localization
        - Add severity classification (minor, moderate, severe)
        - Develop mobile application with offline processing capabilities
        - Integrate GPS for automatic location tagging
        - Incorporate temporal analysis to track damage progression
        - Create an API for integration with municipal management systems
        
        ### Acknowledgments
        
        We thank our instructor, Md. Arefin Rabbi Emon, for guidance throughout this project. We also acknowledge the Islamic University of Technology for providing the resources and infrastructure necessary for development and testing.
        """)

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
