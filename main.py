import streamlit as st
import tensorflow as tf
import numpy as np

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

    # CSS for background image
def add_bg_from_url(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("bg.png");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Dictionaries for Precautions and Treatments
precautions = {
    'Apple___Apple_scab': [
        "Prune infected branches.",
        "Avoid overhead irrigation.",
        "Apply fungicides early in the season."
    ],
    'Apple___Black_rot': [
        "Remove infected fruit and branches.",
        "Maintain proper tree nutrition.",
        "Use resistant apple varieties."
    ],
    'Apple___Cedar_apple_rust': [
        "Remove nearby junipers (hosts of the rust).",
        "Prune out infected leaves.",
        "Use resistant apple varieties."
    ],
    'Apple___healthy': [
        "Ensure proper irrigation.",
        "Regularly check for pests.",
        "Maintain good soil health."
    ],
    'Blueberry___healthy': [
        "Maintain proper soil pH levels.",
        "Ensure regular irrigation.",
        "Monitor for signs of diseases."
    ],
    'Cherry_(including_sour)___Powdery_mildew': [
        "Prune infected branches.",
        "Avoid wetting the foliage.",
        "Apply fungicides like sulfur."
    ],
    'Cherry_(including_sour)___healthy': [
        "Ensure proper spacing between plants.",
        "Monitor soil moisture levels.",
        "Prune regularly to maintain airflow."
    ],
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': [
        "Rotate crops every season.",
        "Use resistant maize varieties.",
        "Apply fungicides if needed."
    ],
    'Corn_(maize)___Common_rust_': [
        "Plant rust-resistant varieties.",
        "Ensure proper field sanitation.",
        "Apply fungicides if rust is detected."
    ],
    'Corn_(maize)___Northern_Leaf_Blight': [
        "Use resistant hybrids.",
        "Ensure good field drainage.",
        "Apply fungicides if necessary."
    ],
    'Corn_(maize)___healthy': [
        "Maintain proper irrigation schedules.",
        "Use pest-resistant corn varieties.",
        "Ensure healthy soil nutrition."
    ],
    'Grape___Black_rot': [
        "Prune out infected leaves and branches.",
        "Improve air circulation by spacing vines properly.",
        "Apply copper-based fungicides."
    ],
    'Grape___Esca_(Black_Measles)': [
        "Remove and destroy infected vines.",
        "Use fungicides sparingly.",
        "Ensure proper irrigation."
    ],
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': [
        "Ensure proper air circulation.",
        "Apply copper-based fungicides.",
        "Regularly inspect vines for symptoms."
    ],
    'Grape___healthy': [
        "Keep vines properly pruned.",
        "Avoid excessive watering.",
        "Monitor soil health regularly."
    ],
    'Orange___Haunglongbing_(Citrus_greening)': [
        "Use disease-free planting material.",
        "Control psyllid insect vectors.",
        "Remove infected trees."
    ],
    'Peach___Bacterial_spot': [
        "Plant resistant varieties.",
        "Avoid overhead irrigation.",
        "Apply copper-based bactericides."
    ],
    'Peach___healthy': [
        "Ensure proper sunlight exposure.",
        "Regularly check for signs of pests.",
        "Maintain healthy soil nutrition."
    ],
    'Pepper,_bell___Bacterial_spot': [
        "Use disease-free seeds and transplants.",
        "Rotate crops regularly.",
        "Avoid working in wet fields to prevent spread."
    ],
    'Pepper,_bell___healthy': [
        "Ensure proper soil drainage.",
        "Maintain good air circulation around plants.",
        "Monitor for signs of pests or diseases."
    ],
    'Potato___Early_blight': [
        "Remove infected plant debris.",
        "Rotate crops with non-host crops.",
        "Apply fungicides if necessary."
    ],
    'Potato___Late_blight': [
        "Use resistant potato varieties.",
        "Ensure proper irrigation management.",
        "Apply fungicides regularly during wet seasons."
    ],
    'Potato___healthy': [
        "Ensure good soil drainage.",
        "Monitor for pest or disease symptoms.",
        "Maintain proper nutrition levels in soil."
    ],
    'Raspberry___healthy': [
        "Prune regularly to maintain airflow.",
        "Monitor for signs of diseases or pests.",
        "Ensure proper irrigation practices."
    ],
    'Soybean___healthy': [
        "Rotate crops regularly.",
        "Ensure proper soil health.",
        "Monitor for signs of diseases or pests."
    ],
    'Squash___Powdery_mildew': [
        "Apply fungicides like sulfur or potassium bicarbonate.",
        "Ensure proper plant spacing for air circulation.",
        "Remove heavily infected leaves."
    ],
    'Strawberry___Leaf_scorch': [
        "Remove infected leaves.",
        "Apply fungicides if necessary.",
        "Avoid overhead watering."
    ],
    'Strawberry___healthy': [
        "Ensure proper sunlight exposure.",
        "Monitor for signs of pests or diseases.",
        "Keep the soil well-drained."
    ],
    'Tomato___Bacterial_spot': [
        "Use disease-free seeds and transplants.",
        "Apply copper-based bactericides.",
        "Ensure proper crop rotation."
    ],
    'Tomato___Early_blight': [
        "Prune lower leaves to prevent infection.",
        "Apply fungicides like chlorothalonil.",
        "Ensure proper crop rotation."
    ],
    'Tomato___Late_blight': [
        "Use resistant varieties.",
        "Apply copper-based fungicides.",
        "Remove and destroy infected plants."
    ],
    'Tomato___Leaf_Mold': [
        "Improve air circulation around plants.",
        "Avoid overhead irrigation.",
        "Apply fungicides if necessary."
    ],
    'Tomato___Septoria_leaf_spot': [
        "Remove and destroy infected leaves.",
        "Apply fungicides like copper or mancozeb.",
        "Ensure proper crop rotation."
    ],
    'Tomato___Spider_mites Two-spotted_spider_mite': [
        "Spray with insecticidal soap or neem oil.",
        "Keep plants well-watered to reduce mite infestations.",
        "Introduce natural predators like ladybugs."
    ],
    'Tomato___Target_Spot': [
        "Remove infected leaves.",
        "Improve air circulation.",
        "Apply fungicides if necessary."
    ],
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': [
        "Use virus-resistant varieties.",
        "Control whiteflies, the insect vector.",
        "Remove and destroy infected plants."
    ],
    'Tomato___Tomato_mosaic_virus': [
        "Use virus-free seeds.",
        "Disinfect tools after use.",
        "Remove infected plants."
    ],
    'Tomato___healthy': [
        "Ensure proper sunlight and irrigation.",
        "Monitor for pests and diseases.",
        "Maintain healthy soil."
    ]
}

treatments = {
    'Apple___Apple_scab': [
        "Apply sulfur-based fungicides.",
        "Use lime-sulfur sprays.",
        "Use organic sprays like neem oil."
    ],
    'Apple___Black_rot': [
        "Use copper-based fungicides.",
        "Remove and destroy infected plant debris.",
        "Spray with a fungicide containing thiophanate-methyl."
    ],
    'Apple___Cedar_apple_rust': [
        "Spray fungicides early in the season.",
        "Remove galls from nearby juniper trees.",
        "Apply a sulfur-based spray during the growing season."
    ],
    'Apple___healthy': [
        "No treatment required for healthy plants.",
        "Continue monitoring regularly.",
        "Maintain good cultural practices."
    ],
    'Blueberry___healthy': [
        "No specific treatment needed.",
        "Ensure proper care and maintenance.",
        "Regularly inspect for early signs of diseases."
    ],
    'Cherry_(including_sour)___Powdery_mildew': [
        "Apply fungicides like sulfur or potassium bicarbonate.",
        "Prune infected leaves and branches.",
        "Avoid overhead watering."
    ],
    'Cherry_(including_sour)___healthy': [
        "No treatment required for healthy plants.",
        "Regularly monitor and maintain good practices.",
        "Prune regularly to ensure good airflow."
    ],
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': [
        "Apply fungicides like strobilurins.",
        "Remove infected plant debris.",
        "Rotate crops annually to prevent recurrence."
    ],
    'Corn_(maize)___Common_rust_': [
        "Apply fungicides containing mancozeb.",
        "Plant resistant corn varieties.",
        "Ensure proper air circulation in the field."
    ],
    'Corn_(maize)___Northern_Leaf_Blight': [
        "Apply fungicides like azoxystrobin.",
        "Use disease-resistant hybrids.",
        "Rotate crops to prevent pathogen buildup."
    ],
    'Corn_(maize)___healthy': [
        "No treatment needed for healthy plants.",
        "Ensure proper care and regular inspections.",
        "Continue maintaining soil health."
    ],
    'Grape___Black_rot': [
        "Apply fungicides like mancozeb or myclobutanil.",
        "Prune and destroy infected plant debris.",
        "Maintain good airflow through pruning."
    ],
    'Grape___Esca_(Black_Measles)': [
        "Remove and destroy severely infected vines.",
        "Apply fungicides sparingly and only as a preventive measure.",
        "Maintain proper irrigation to avoid over-watering."
    ],
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': [
        "Apply fungicides such as copper-based sprays.",
        "Prune vines to increase air circulation.",
        "Remove infected leaves to prevent spread."
    ],
    'Grape___healthy': [
        "No treatment required for healthy vines.",
        "Continue regular pruning and monitoring.",
        "Maintain good soil nutrition."
    ],
    'Orange___Haunglongbing_(Citrus_greening)': [
        "Remove infected trees.",
        "Apply insecticides to control psyllid populations.",
        "Use disease-free planting material."
    ],
    'Peach___Bacterial_spot': [
        "Apply copper-based bactericides.",
        "Remove and destroy infected leaves and fruits.",
        "Use resistant peach varieties."
    ],
    'Peach___healthy': [
        "No treatment required for healthy trees.",
        "Continue monitoring for signs of diseases.",
        "Maintain proper irrigation and pruning practices."
    ],
    'Pepper,_bell___Bacterial_spot': [
        "Apply copper-based sprays.",
        "Rotate crops regularly.",
        "Remove and destroy infected plants."
    ],
    'Pepper,_bell___healthy': [
        "No treatment required for healthy plants.",
        "Maintain regular monitoring and care.",
        "Ensure proper soil drainage and air circulation."
    ],
    'Potato___Early_blight': [
        "Apply fungicides like chlorothalonil or mancozeb.",
        "Remove and destroy infected plant debris.",
        "Rotate crops to prevent pathogen buildup."
    ],
    'Potato___Late_blight': [
        "Apply fungicides like copper or mancozeb.",
        "Remove and destroy infected plants.",
        "Ensure proper soil drainage and avoid over-watering."
    ],
    'Potato___healthy': [
        "No treatment required for healthy plants.",
        "Continue regular monitoring and care.",
        "Maintain proper soil moisture and nutrition."
    ],
    'Raspberry___healthy': [
        "No treatment required for healthy plants.",
        "Regularly monitor and maintain good practices.",
        "Ensure proper air circulation and irrigation."
    ],
    'Soybean___healthy': [
        "No treatment required for healthy crops.",
        "Continue regular monitoring and care.",
        "Rotate crops to maintain soil health."
    ],
    'Squash___Powdery_mildew': [
        "Apply sulfur or neem oil sprays.",
        "Prune and remove heavily infected leaves.",
        "Ensure proper air circulation between plants."
    ],
    'Strawberry___Leaf_scorch': [
        "Apply fungicides like myclobutanil.",
        "Remove and destroy infected plant debris.",
        "Avoid overhead watering to reduce humidity."
    ],
    'Strawberry___healthy': [
        "No treatment required for healthy plants.",
        "Maintain regular care and monitoring.",
        "Ensure proper soil drainage and irrigation."
    ],
    'Tomato___Bacterial_spot': [
        "Apply copper-based bactericides.",
        "Remove and destroy infected plant debris.",
        "Use disease-free seeds and transplants."
    ],
    'Tomato___Early_blight': [
        "Apply fungicides like chlorothalonil.",
        "Prune lower leaves to reduce humidity.",
        "Rotate crops to prevent pathogen buildup."
    ],
    'Tomato___Late_blight': [
        "Apply fungicides like copper or mancozeb.",
        "Remove and destroy infected plants.",
        "Ensure good air circulation and avoid over-watering."
    ],
    'Tomato___Leaf_Mold': [
        "Apply fungicides like chlorothalonil or mancozeb.",
        "Remove and destroy infected leaves.",
        "Ensure good air circulation between plants."
    ],
    'Tomato___Septoria_leaf_spot': [
        "Apply fungicides like mancozeb or copper.",
        "Remove and destroy infected leaves.",
        "Ensure proper crop rotation to prevent recurrence."
    ],
    'Tomato___Spider_mites Two-spotted_spider_mite': [
        "Apply insecticidal soap or neem oil.",
        "Keep plants well-watered to reduce mite infestations.",
        "Introduce natural predators like ladybugs."
    ],
    'Tomato___Target_Spot': [
        "Apply fungicides like chlorothalonil or copper.",
        "Remove and destroy infected leaves.",
        "Improve air circulation by spacing plants properly."
    ],
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': [
        "Remove and destroy infected plants.",
        "Control whiteflies using insecticidal sprays.",
        "Use virus-resistant tomato varieties."
    ],
    'Tomato___Tomato_mosaic_virus': [
        "Remove and destroy infected plants.",
        "Disinfect tools and equipment regularly.",
        "Use virus-free seeds."
    ],
    'Tomato___healthy': [
        "No treatment required for healthy plants.",
        "Maintain regular monitoring and proper care.",
        "Ensure proper soil nutrition and air circulation."
    ]
}


# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Set background image (change URL as needed for the desired image)
background_image_url = "bg.png"  # Replace with your image URL
add_bg_from_url(background_image_url)

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.
    
    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.
    
    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset consists of about 87K RGB images of healthy and diseased crop leaves categorized into 38 different classes.
    
    #### Content
    1. Train (70,295 images)
    2. Test (33 images)
    3. Validation (17,572 images)
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    
    if st.button("Show Image"):
        st.image(test_image, width=4, use_column_width=True)
    
    # Predict button
    if st.button("Predict"):
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        
        # Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                      'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                      'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                      'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                      'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                      'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                      'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                      'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                      'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                      'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                      'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                      'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                      'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']

        predicted_class = class_name[result_index]
        st.success(f"Model is Predicting it's a {predicted_class}")

        # Display Precautions and Treatments if available
        if predicted_class in precautions:
            st.subheader("Precautions:")
            for precaution in precautions[predicted_class]:
                st.write(f"- {precaution}")
        
        if predicted_class in treatments:
            st.subheader("Treatments:")
            for treatment in treatments[predicted_class]:
                st.write(f"- {treatment}")
