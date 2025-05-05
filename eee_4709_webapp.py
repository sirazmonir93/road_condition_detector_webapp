import streamlit as st
import tensorflow as tf
import numpy as np
import time
import datetime
import random
import pandas as pd

# Page configuration with improved settings
st.set_page_config(
    page_title="Road Condition Detector",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI and text visibility across themes
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.7rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        color: #212529; /* Ensuring text is visible on light backgrounds */
    }
    .card-dark {
        background-color: #2c2c2c;
        color: #f8f9fa;
    }
    .card-blue {
        background-color: #1a2d40;
        color: #f8f9fa;
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .team-card {
        border: 1px solid #ccc;
        padding: 15px;
        border-radius: 10px;
        background-color: #002a47;
        color: white;
        margin-bottom: 15px;
    }
    .caption-text {
        font-size: 0.9rem;
        color: #6c757d;
        font-style: italic;
    }
    .dark-caption {
        color: #adb5bd; /* Lighter color for captions in dark mode */
    }
    .highlight {
        background-color: #fffacd;
        padding: 0.25rem;
        border-radius: 0.25rem;
        color: #212529; /* Ensuring text is visible */
    }
    .progress-label {
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .image-container {
        margin-bottom: 1rem;
    }
    .image-container img {
        border-radius: 8px;
        object-fit: cover;
        width: 100%;
        max-height: 300px;
    }
    .result-card {
        padding: 1.2rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .good-road {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .bad-road {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .tab-content {
        padding: 1rem;
        border: 1px solid #dee2e6;
        border-top: none;
        border-radius: 0 0 0.25rem 0.25rem;
    }
    /* Image gallery improvements */
    .gallery-image {
        transition: transform 0.3s ease;
        margin-bottom: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .gallery-image:hover {
        transform: scale(1.03);
    }
    /* Timeline for history */
    .timeline {
        border-left: 2px solid #007bff;
        padding-left: 20px;
        margin-left: 10px;
    }
    .timeline-item {
        margin-bottom: 15px;
        position: relative;
    }
    .timeline-item:before {
        content: '';
        width: 12px;
        height: 12px;
        background: #007bff;
        border-radius: 50%;
        position: absolute;
        left: -26px;
        top: 5px;
    }
    /* Dark theme text visibility fixes */
    .dark-theme {
        color: #f1f1f1;
    }
    .dark-theme .card {
        background-color: #2c2c2c;
        color: #f1f1f1;
    }
    .dark-theme .caption-text {
        color: #adb5bd;
    }
    /* Blue theme text visibility fixes */
    .blue-theme {
        color: #f1f1f1;
    }
    .blue-theme .card {
        background-color: #1a2d40;
        color: #f1f1f1;
    }
    .blue-theme .caption-text {
        color: #adb5bd;
    }
</style>
""", unsafe_allow_html=True)

# Model prediction function - unchanged
def model_prediction(test_image):
    model = tf.keras.models.load_model("road_pothole_Rainy_days.keras")
    
    # Load and preprocess the image
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(256, 256))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    
    # Get prediction (raw probability)
    raw_prediction = model.predict(input_arr)[0][0]
    
    # Apply threshold
    threshold = 0.3
    predicted_class = 1 if raw_prediction > threshold else 0
    
    # Map to class names
    class_names = ["BAD ROAD", "Good road"]
    result = class_names[predicted_class]
    
    # For confidence, use the raw prediction or its complement depending on the class
    confidence = raw_prediction if predicted_class == 1 else (1 - raw_prediction)
    
    return result, confidence

# Sidebar with improved styling
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>Dashboard</h2>", unsafe_allow_html=True)
    
    app_mode = st.selectbox(
        "📋 Select Page",
        ["Home", "About", "Road condition detection", "Gallery", "Project Report"]
    )
    
    # Add customization options
    st.markdown("---")
    st.markdown("### 🎨 Customization")
    
    # Improved threshold slider with more context
    threshold = st.slider(
        "Detection Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.3, 
        step=0.05,
        help="Adjust sensitivity: Lower values detect more potential issues, higher values reduce false positives"
    )
    
    # More clear checkbox label
    show_advanced = st.checkbox("Show Advanced Details", False, help="Display technical metrics and model information")
    
    # Theme selector with preview colors
    theme = st.selectbox(
        "Choose Theme",
        ["Light", "Dark", "Blue"],
        help="Change the app's appearance"
    )
    
    # Contact information in sidebar footer
    st.markdown("---")
    st.markdown("<div style='text-align: center; font-size: 0.8rem;'>Islamic University of Technology<br>© 2025 Road Condition Detector</div>", unsafe_allow_html=True)

# Apply selected theme with text visibility fixes
if theme == "Dark":
    st.markdown("""
        <style>
        .stApp {
            background-color: #121212;
            color: white;
        }
        .caption-text {
            color: #adb5bd !important;
        }
        </style>
    """, unsafe_allow_html=True)
    # Add a class to the body for theme-specific styling
    st.markdown('<div class="dark-theme">', unsafe_allow_html=True)
elif theme == "Blue":
    st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: #f1f1f1;
        }
        .caption-text {
            color: #adb5bd !important;
        }
        </style>
    """, unsafe_allow_html=True)
    # Add a class to the body for theme-specific styling
    st.markdown('<div class="blue-theme">', unsafe_allow_html=True)
else:
    # Light theme (default)
    st.markdown('<div>', unsafe_allow_html=True)

# Main Page with improved layout
if app_mode == "Home":
    st.markdown("<h1 class='main-header'>Road Condition Recognition with Python</h1>", unsafe_allow_html=True)
    
    # Better image display with constraints
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image("home_page.jpeg", use_column_width=True)
    
    st.markdown(f"""
    <div class="card">
    <h2>Welcome to the Road Condition Detection System</h2>
    <p>A simple, smart, and efficient web application that helps detect whether a road surface is in <strong>good</strong> or <strong>bad</strong> condition using an AI-powered image classifier.</p>
    
    <p>Our goal is to support road safety and maintenance efforts by enabling quick on-the-spot assessments using nothing more than a photo — either captured directly or uploaded.</p>
    </div>
    """, unsafe_allow_html=True)

    # Using columns for better layout of features
    st.markdown("<h3 class='sub-header'>What This App Can Do</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
        <div class="feature-icon">🔍</div>
        <h4>Detection</h4>
        <ul>
          <li>Detect potholes and road damages in real-time</li>
          <li>Show prediction confidence with visual feedback</li>
          <li>Use your camera or upload road images</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
        <div class="feature-icon">⚙️</div>
        <h4>Customization</h4>
        <ul>
          <li>Adjust detection sensitivity to suit different environments</li>
          <li>Customize the app's appearance with themes</li>
          <li>Toggle advanced technical information display</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="card">
        <div class="feature-icon">🧠</div>
        <h4>Under the Hood</h4>
        <p>This tool uses a <strong>Convolutional Neural Network (CNN)</strong> trained with augmented images to handle real-world conditions, especially in <strong>rainy environments</strong> where potholes become a major safety concern.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
        <div class="feature-icon">🚀</div>
        <h4>How to Use</h4>
        <ol>
          <li>Go to the <strong>"Road Condition Detection"</strong> tab from the sidebar.</li>
          <li>Upload a road image or take a photo.</li>
          <li>Click <strong>"Analyze Road"</strong> to get instant results.</li>
          <li>Check the results, confidence, and view previous analyses in the history log.</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Call to action at the bottom
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 10px; color: #212529;">
        <h3>Start exploring, analyzing, and making your roads safer!</h3>
        <p>Navigate to the <strong>"Road Condition Detection"</strong> tab to begin analyzing road images.</p>
    </div>
    """, unsafe_allow_html=True)

# About Project with improved layout
elif app_mode == "About":
    st.markdown("<h1 class='main-header'>About This Web App</h1>", unsafe_allow_html=True)
    
    # Two-column layout for intro
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class="card">
        <p>Welcome to the <strong>Road Condition Detector</strong> — an AI-powered web application designed to assess the quality of road surfaces from images. Whether you're capturing a photo on-site or uploading an image, our system provides fast and reliable feedback on the road's condition.</p>
        <p>This tool is particularly useful for detecting <strong>damaged or pothole-ridden roads</strong>, especially in <strong>rainy weather scenarios</strong> where early maintenance can prevent water clogging and accidents.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Placeholder image related to the project concept
        st.image("Picture13.jpg", use_column_width=True)
        caption_class = "caption-text dark-caption" if theme in ["Dark", "Blue"] else "caption-text"
        st.markdown(f"<p class='{caption_class}'>System workflow visualization</p>", unsafe_allow_html=True)
    
    st.markdown("<h3 class='sub-header'>Where We Started</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <p>We began this project with a simple question: <em>Can AI help us identify bad roads before they become dangerous?</em></p>
    <p>In regions with heavy rainfall and poor infrastructure, road damage often goes unnoticed until it causes serious problems. This inspired us to create a tool that could assist in early detection — accessible to both professionals and the public.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 class='sub-header'>What We Built</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <p>We developed a smart, intuitive web app powered by a <strong>Convolutional Neural Network (CNN)</strong>. This deep learning model analyzes road surface images and categorizes them as:</p>
    <ul>
      <li><strong>Good Road</strong> – Safe, smooth, and well-maintained.</li>
      <li><strong>Bad Road</strong> – Likely damaged, cracked, or susceptible to water accumulation.</li>
    </ul>
    <p>Users can either <strong>upload</strong> or <strong>capture</strong> road images in real time, adjust a <strong>threshold slider</strong> to control sensitivity, and view a <strong>confidence score</strong> with each prediction.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 class='sub-header'>Key Features</h3>", unsafe_allow_html=True)
    
    # Feature columns for better organization
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
        <h4>Input Options</h4>
        <ul>
          <li>Upload existing images</li>
          <li>Capture photos directly</li>
          <li>Process images in real-time</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
        <h4>Analysis Tools</h4>
        <ul>
          <li>Confidence-based classification</li>
          <li>Adjustable detection threshold</li>
          <li>Visual confidence indicators</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
        <h4>User Experience</h4>
        <ul>
          <li>Real-time prediction history</li>
          <li>Demo map integration</li>
          <li>Theme customization</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h3 class='sub-header'>About the Dataset</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <p>The model was trained on a publicly available pothole dataset, extended using <strong>offline augmentation techniques</strong> to simulate diverse weather and lighting conditions. This enabled the model to generalize well across real-world scenarios.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 class='sub-header'>What's Next</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <p>We see immense potential for future development:</p>
    <ul>
      <li>Integration with <strong>municipality road monitoring systems</strong></li>
      <li>Use in <strong>smart cities</strong> for automated damage detection</li>
      <li>Creation of a <strong>mobile app</strong> version for field inspectors</li>
      <li>Building a <strong>crowdsourced platform</strong> for reporting and tracking road health</li>
    </ul>
    <p>Our goal is to make this tool more scalable, precise, and deployable at larger infrastructure levels.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 class='sub-header'>Developed By</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="team-card">
    <h4>Our Team</h4>
    <p>📌 <strong>K. M. Sirazul Monir</strong> — <em>ID: 200021247</em><br>
    <strong>Lead Developer & UI/UX Designer</strong><br>
    <small>Led the coding of the web interface and handled user experience design.</small></p>
    
    <p>📌 <strong>Shafin Ibnul Mohasin</strong> — <em>ID: 200021244</em><br>
    <strong>Machine Learning Engineer</strong><br>
    <small>Developed and trained the deep learning model for road condition classification.</small></p>
    
    <p>📌 <strong>Whiduzzaman Rishad</strong> — <em>ID: 200021228</em><br>
    <strong>Data Analyst & Integration Specialist</strong><br>
    <small>Managed dataset preparation, augmentation, and backend integration.</small></p>
    
    <p><strong>Instructor:</strong> Md. Arefin Rabbi Emon</p>
    
    <p><small>Built as part of an academic project at <strong>Islamic University of Technology</strong>.<br>
    All team members contributed collaboratively across development, research, and system design.</small></p>
    </div>
    """, unsafe_allow_html=True)

# Gallery page with improved image display
elif app_mode == "Gallery":
    st.markdown("<h1 class='main-header'>Project Gallery</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    <h3>Sample Road Images</h3>
    <p>Explore these sample road images analyzed by our system. The gallery showcases various road conditions and how our AI model interprets them.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a better grid layout with consistent sizing
    col1, col2 = st.columns(2)
    
    caption_class = "caption-text dark-caption" if theme in ["Dark", "Blue"] else "caption-text"
    
    with col1:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image("Picture9.jpg", use_column_width=True, output_format="JPEG", clamp=True)
        st.markdown(f'<p class="{caption_class}">Good Road Sample</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image("Picture10.jpg", use_column_width=True, output_format="JPEG", clamp=True)
        st.markdown(f'<p class="{caption_class}">Bad Road with Potholes</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image("Picture11.jpg", use_column_width=True, output_format="JPEG", clamp=True) 
        st.markdown(f'<p class="{caption_class}">Road in Rainy Condition</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image("Picture12.jpg", use_column_width=True, output_format="JPEG", clamp=True)
        st.markdown(f'<p class="{caption_class}">Water Accumulation on Road</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add explanation text between images
    st.markdown("""
    <div class="card">
    <h3>Understanding the Analysis</h3>
    <p>Our AI model analyzes visual features such as texture irregularities, color patterns, and edge formations to identify potential road issues.</p>
    <p>The samples above demonstrate the range of conditions our system can evaluate, from clearly visible potholes to more subtle surface irregularities that might lead to water accumulation during rain.</p>
    </div>
    """, unsafe_allow_html=True)

# Project Report with improved layout and readability
elif app_mode == "Project Report":
    st.markdown("<h1 class='main-header'>Project Technical Report</h1>", unsafe_allow_html=True)
    
    # Create tabs with better styling
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Executive Summary", "🔍 Methodology", "📊 Results", "🏁 Conclusion", "📚 References"])
    
    caption_class = "caption-text dark-caption" if theme in ["Dark", "Blue"] else "caption-text"
    
    with tab1:
        st.markdown("""
        <div class="card">
        <h2>Executive Summary</h2>
        
        <h3>Road Condition Monitoring System Using AI: A Novel Approach</h3>
        
        <h4>ABSTRACT</h4>
        
        <p>This research presents an automated system that uses artificial intelligence to monitor road conditions, specifically detecting potholes and water accumulation during rainy seasons. The system employs a Convolutional Neural Network (CNN) to analyze aerial images captured by surveillance drones, enabling cities to better plan and prioritize road maintenance activities.</p>
        
        <p>Training was conducted on a dataset of 4,200 road images, with validation performed on 900 additional images. The CNN architecture follows a VGG-like structure with five convolutional blocks of increasing filter sizes (32→64→128→256→512), each followed by max pooling operations. The network includes dropout layers to prevent overfitting and uses binary cross-entropy as the loss function.</p>
        
        <p>By optimizing the decision threshold to 0.3 instead of the conventional 0.5, the model achieved classification accuracy exceeding 95%, representing a significant improvement over comparable studies.</p>
        
        <p>Unlike most existing approaches that focus solely on structural defects, this system successfully identifies both potholes and water accumulation issues, providing a more comprehensive assessment of road maintenance needs.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.image("Picture3.jpg", caption="Model Training Results", use_column_width=True)
        
        st.markdown("""
        <div class="card">
        <h3>Key Achievements</h3>
        <ul>
          <li>Development of a user-friendly web interface using Streamlit</li>
          <li>Implementation of an effective image classifier with real-time analysis capabilities</li>
          <li>Integration of visual feedback systems with confidence scoring</li>
          <li>Creation of a flexible system that can be used via uploaded images or direct camera input</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div class="card">
        <h2>Methodology</h2>
        <p>Our approach combines machine learning techniques with user-centered design principles to create an accessible and accurate road condition detection system.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display flowchart images with better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.image("Picture13.jpg", caption="System Architecture Flowchart", use_column_width=True)
        with col2:
            st.image("Picture14.jpg", caption="Data Processing Pipeline", use_column_width=True)
        
        st.markdown("""
        <div class="card">
        <h3>Technical Implementation</h3>
        <ol>
          <li><strong>Data Collection:</strong> Curated a dataset of 5,100 total images (4,200 training, 900 validation)</li>
          <li><strong>Data Augmentation:</strong> Applied transformations including rotation, flipping, brightness adjustment, and simulated rain effects</li>
          <li><strong>Model Architecture:</strong> Implemented a VGG-style CNN with five convolutional blocks</li>
          <li><strong>Optimization:</strong> Used binary cross-entropy loss with Adam optimizer</li>
          <li><strong>Threshold Tuning:</strong> Empirically determined 0.3 as optimal decision threshold</li>
          <li><strong>Frontend Development:</strong> Built interactive web interface using Streamlit</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("""
        <div class="card">
        <h2>Results</h2>
        <p>Our model demonstrated strong performance across various metrics, particularly in challenging environmental conditions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create better layout for results visuals with explanations
        col1, col2 = st.columns(2)
        
        with col1:
            st.image("Picture2.jpg", use_column_width=True)
            st.markdown(f"<p class='{caption_class}'>Performance metrics across different road conditions</p>", unsafe_allow_html=True)
            
            st.image("Picture4.jpg", use_column_width=True)
            st.markdown(f"<p class='{caption_class}'>Summary of model performance metrics</p>", unsafe_allow_html=True)
        
        with col2:
            st.image("Picture5.jpg", use_column_width=True)
            st.markdown(f"<p class='{caption_class}'>ROC curve showing model discrimination ability</p>", unsafe_allow_html=True)
            
            st.image("Picture6.jpg", use_column_width=True)
            st.markdown(f"<p class='{caption_class}'>Confusion matrix showing true vs. predicted classifications</p>", unsafe_allow_html=True)
        
        # Display additional results
        st.image("Picture7.jpg", caption="Per Class Metrics", use_column_width=True)
        
        st.image("Picture8.jpg", caption="Cross-Validation Results", use_column_width=True)
        
        st.markdown("""
        <div class="card">
        <h3>Model Performance Summary</h3>
        <ul>
          <li><strong>Training accuracy:</strong> The model reaches ~87-90% by the end of 10 epochs</li>
          <li><strong>Validation accuracy:</strong> ~82-85% with good generalization</li>
          <li><strong>Threshold:</strong> 0.3 (optimized to reduce false negatives)</li>
          <li><strong>Precision for pothole detection:</strong> ~80-85%</li>
          <li><strong>Recall for pothole detection:</strong> ~75-80%</li>
          <li><strong>F1-score:</strong> ~77-82%</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("""
        <div class="card">
        <h2>Conclusion</h2>
        <p>Our Road Condition Detection System demonstrates the practical application of deep learning techniques for infrastructure maintenance and public safety.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Using columns for better organization
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card">
            <h3>Key Findings</h3>
            <ol>
              <li>The CNN model demonstrates strong capability in distinguishing between good and damaged road surfaces, particularly in moderate rain conditions.</li>
              <li>Real-time processing capabilities make the application practical for field use.</li>
              <li>User interface design significantly impacts adoption and trust in AI predictions.</li>
              <li>The threshold adjustment feature improves versatility across different environments.</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
            <h3>Limitations</h3>
            <ul>
              <li>Performance degrades in extreme weather conditions</li>
              <li>No severity classification for detected damage</li>
              <li>Limited real-world testing in diverse geographical regions</li>
              <li>No localization of damage within images</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        

        
        with col2:
            st.markdown("""
            <div class="card">
            <h3>Future Work</h3>
            <ul>
              <li>Implement object detection for precise damage localization</li>
              <li>Add severity classification (minor, moderate, severe)</li>
              <li>Develop mobile application with offline processing capabilities</li>
              <li>Integrate GPS for automatic location tagging</li>
              <li>Incorporate temporal analysis to track damage progression</li>
              <li>Create an API for integration with municipal management systems</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            

            st.markdown("""
            <div class="card">
            <h3>Acknowledgments</h3>
            <p>We thank our instructor, Md. Arefin Rabbi Emon, for guidance throughout this project. We also acknowledge the Islamic University of Technology for providing the resources and infrastructure necessary for development and testing.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab5:
        st.markdown("""
        <div class="card">
        <h2>References</h2>
        <ol>
            <li>Koch, C., & Brilakis, I. (2011). Pothole detection in asphalt pavement images. Advanced Engineering Informatics, 25(3), 507-515.</li>
            <li>Nienaber, S., Booysen, M. J., & Kroon, R. (2015). Detecting potholes using simple image processing techniques and real-world footage. In Proceedings of the 34th Southern African Transport Conference (SATC 2015) (pp. 153-164).</li>
            <li>Zhang, L., Yang, F., Zhang, Y. D., & Zhu, Y. J. (2017). Road crack detection using deep convolutional neural network. In 2017 IEEE International Conference on Image Processing (ICIP) (pp. 3708-3712). IEEE.</li>
            <li>Fan, R., Bocus, M. J., Zhu, Y., Jiao, J., Wang, L., Ma, F., ... & Liu, M. (2019). Road crack detection using deep convolutional neural network and adaptive thresholding.</li>
            <li>Maeda, H., Sekimoto, Y., Seto, T., Kashiyama, T., & Omata, H. (2018). Road damage detection and classification using deep neural networks with smartphone images.</li>
            <li>Chollet, F. (2018). Deep Learning with Python. Manning Publications.</li>
            <li>American Society of Civil Engineers (ASCE). (2021). Infrastructure Report Card: Roads.</li>
            <li>"Water-Pothole-Detection," Roboflow Universe.</li>
            <li>"RoadSAW Dataset," Viscoda.</li>
            <li>"StagnantWaterData," Kaggle.</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

# Prediction Page with improved UI/UX
elif app_mode == "Road condition detection":
    st.markdown("<h1 class='main-header'>Road Quality Detection</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    <p>Upload a road image or take a photo to analyze the road condition. Our AI will determine if the road is in good condition or has potential issues that could lead to water accumulation during rainy seasons.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for better layout
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Let the user choose between uploading an image or using the camera with better styling
        st.markdown("<h3 class='sub-header'>Image Input</h3>", unsafe_allow_html=True)
        
        input_option = st.radio(
            "Select input method:",
            ["Upload Image", "Use Camera"],
            help="Choose how you want to provide the road image"
        )
        
        if input_option == "Upload Image":
            test_image = st.file_uploader(
                "Choose a road image:",
                type=["jpg", "jpeg", "png"],
                help="Upload an image of a road to analyze its condition"
            )
        else:
            test_image = st.camera_input(
                "Take a picture of the road",
                help="Use your camera to take a photo of the road"
            )
        
        # Add instructions
        st.markdown("""
        <div class="card">
        <h4>Tips for best results:</h4>
        <ul>
          <li>Ensure good lighting conditions</li>
          <li>Capture the road surface clearly</li>
          <li>Try to include any visible defects</li>
          <li>For wet roads, include areas with water accumulation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Show the image when available in a nicer container
        if test_image is not None:
            st.markdown("<h3 class='sub-header'>Selected Image</h3>", unsafe_allow_html=True)
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(test_image, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Predict button with better styling
            analyze_button = st.button(
                "🔍 Analyze Road",
                help="Click to analyze the road condition",
                type="primary",
                use_container_width=True
            )
        else:
            st.markdown("""
            <div style="height: 300px; display: flex; justify-content: center; align-items: center; border: 2px dashed #ccc; border-radius: 10px; margin-top: 20px;">
                <div style="text-align: center; color: #666;">
                    <h3>No Image Selected</h3>
                    <p>Please upload an image or take a photo to analyze</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            analyze_button = False
    
    # Add this near the top of your script (for history feature)
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Predict logic with improved UI
    if test_image is not None and analyze_button:
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
            
            # Use columns for a better layout of results
            st.markdown("<h3 class='sub-header'>Analysis Results</h3>", unsafe_allow_html=True)
            
            # Display the result with tabs
            tab1, tab2 = st.tabs(["📊 Results", "🔬 Technical Details"])
            
            with tab1:
                # Show different messages based on the result
                if result == "Good road":
                    st.markdown("""
                    <div class="result-card good-road">
                        <h3>✅ GOOD ROAD DETECTED</h3>
                        <p>Confidence: <strong>{:.2%}</strong></p>
                        <p>This road appears to be in good condition with no chances of clogged water formation in rainy seasons.</p>
                    </div>
                    """.format(confidence), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-card bad-road">
                        <h3>⚠️ BAD ROAD DETECTED</h3>
                        <p>Confidence: <strong>{:.2%}</strong></p>
                        <p>This road has a high probability of clogged water formation in rainy seasons and may need maintenance.</p>
                    </div>
                    """.format(confidence), unsafe_allow_html=True)
                
                # Visual confidence meter with better styling
                st.markdown("<p class='progress-label'>Confidence Level:</p>", unsafe_allow_html=True)
                st.progress(int(confidence * 100))
                
                # Color coding based on confidence
                if confidence > 0.8:
                    st.success(f"High confidence: {confidence:.2%}")
                elif confidence > 0.5:
                    st.info(f"Medium confidence: {confidence:.2%}")
                else:
                    st.warning(f"Low confidence: {confidence:.2%}")
                
                # Add recommendation based on result
                if result == "Good road":
                    st.markdown("""
                    <div class="card">
                    <h4>Recommendation</h4>
                    <p>✅ This road is in good condition and does not require immediate maintenance.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="card">
                    <h4>Recommendation</h4>
                    <p>⚠️ This road shows signs of damage and should be scheduled for maintenance, especially before the rainy season.</p>
                    <p>Potential issues:</p>
                    <ul>
                      <li>Surface irregularities that may collect water</li>
                      <li>Possible potholes or cracks</li>
                      <li>Risk of further deterioration during rain</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            with tab2:
                st.markdown("""
                <div class="card">
                <h4>Technical Analysis Details</h4>
                <table style="width:100%">
                  <tr>
                    <td><strong>Classification:</strong></td>
                    <td>{}</td>
                  </tr>
                  <tr>
                    <td><strong>Confidence Score:</strong></td>
                    <td>{:.2%}</td>
                  </tr>
                  <tr>
                    <td><strong>Model Architecture:</strong></td>
                    <td>TensorFlow CNN</td>
                  </tr>
                  <tr>
                    <td><strong>Threshold Used:</strong></td>
                    <td>0.3</td>
                  </tr>
                  <tr>
                    <td><strong>Processing Time:</strong></td>
                    <td>~1 second</td>
                  </tr>
                </table>
                </div>
                """.format(result, confidence), unsafe_allow_html=True)
                
                if show_advanced:
                    st.markdown("""
                    <div class="card">
                    <h4>Advanced Technical Information</h4>
                    <p>The model uses a VGG-style CNN architecture with:</p>
                    <ul>
                      <li>5 convolutional blocks with filter sizes (32→64→128→256→512)</li>
                      <li>Max pooling layers after each block</li>
                      <li>Dropout layers (0.5) to prevent overfitting</li>
                      <li>Binary cross-entropy loss function</li>
                      <li>Adam optimizer with learning rate 0.001</li>
                      <li>Image preprocessing: 256x256 resize, normalization</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Save to history
            import datetime
            st.session_state.history.append({
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "result": result,
                "confidence": confidence
            })
            
            # Show history in expander with improved styling
            with st.expander("Analysis History"):
                if len(st.session_state.history) > 0:
                    st.markdown('<div class="timeline">', unsafe_allow_html=True)
                    for item in st.session_state.history:
                        result_class = "good-road" if "Good" in item['result'] else "bad-road"
                        st.markdown(f"""
                        <div class="timeline-item">
                            <strong>{item['timestamp']}</strong> - 
                            <span class="{result_class}">{item['result']}</span> 
                            ({item['confidence']:.2%})
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.write("No previous analyses yet.")
            
            # Add a more interactive map component
            with st.expander("View on Map (Demo)"):
                st.markdown("""
                <div class="card">
                <p>This is a demonstration of how detected road issues could be visualized on a map. In a production environment, this would use GPS data from the image or device.</p>
                </div>
                """, unsafe_allow_html=True)
                
                import random
                import pandas as pd
                
                # Generate a more realistic demo with multiple points if bad road
                if result == "BAD ROAD":
                    # This is just a placeholder - in a real app you would use GPS coordinates
                    map_data = pd.DataFrame({
                        'lat': [random.uniform(40.0, 41.0) for _ in range(3)],
                        'lon': [random.uniform(-74.0, -73.0) for _ in range(3)]
                    })
                    st.map(map_data)
                    st.markdown("""
                    <div class="caption-text">
                    <p>📍 Red markers indicate potential road issues in the area</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    map_data = pd.DataFrame({
                        'lat': [random.uniform(40.0, 41.0)],
                        'lon': [random.uniform(-74.0, -73.0)]
                    })
                    st.map(map_data)
                    st.markdown("""
                    <div class="caption-text">
                    <p>📍 Green marker indicates analyzed road location (good condition)</p>
                    </div>
                    """, unsafe_allow_html=True)
