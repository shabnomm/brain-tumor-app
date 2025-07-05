import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# App settings
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .title {text-align: center;}
    .prediction-card {
        background-color: #f0f2f6;
        border-radius: 20px;
        padding: 30px;
        margin-top: 30px;
        text-align: center;
        box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.1);
    }
    .confidence-bar > div {
        background-color: #6c63ff !important;
    }
    .tumor-label {
        font-size: 26px;
        font-weight: bold;
        color: #6c63ff;
    }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown("<h1 class='title'>🧠 Brain Tumor Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload an MRI image to detect the type of brain tumor with AI.</p>", unsafe_allow_html=True)

# Load trained model
model = load_model("brain_tumor_model.keras")
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Upload image
uploaded_file = st.file_uploader("📤 Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        with st.spinner("🔍 Analyzing the MRI image..."):
            # Load and display image
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="🖼️ Uploaded MRI Image", use_column_width=True)

            # Preprocess
            img = img.resize((150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Predict
            prediction = model.predict(img_array)[0]
            predicted_class = class_names[np.argmax(prediction)]
            confidence = round(100 * np.max(prediction), 2)

        # Display result in styled card
        st.markdown(f"""
            <div class="prediction-card">
                <div class="tumor-label">🔬 Predicted: {predicted_class.upper()}</div>
                <p>Confidence Score:</p>
                <div class="confidence-bar">
                    <progress value="{confidence}" max="100" style="width: 80%; height: 20px;"></progress>
                    <p><strong>{confidence}%</strong></p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Class explanations
        descriptions = {
            'glioma': 'Gliomas are tumors that originate in the glial cells of the brain.',
            'meningioma': 'Meningiomas arise from the protective membranes (meninges) surrounding the brain.',
            'no_tumor': 'No brain tumor detected in the uploaded scan.',
            'pituitary': 'Pituitary tumors develop in the pituitary gland, which controls hormone production.'
        }
        st.info(f"📘 **About the prediction:** {descriptions.get(predicted_class, 'No details available.')}")
        
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
else:
    st.markdown("<br><i>Awaiting MRI image upload...</i>", unsafe_allow_html=True)
