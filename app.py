import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps
from fpdf import FPDF
import tempfile
import os
import datetime

st.set_page_config(page_title="Alzheimer Detection", layout="wide")

st.markdown("""
<style>
    .stApp {background-color: #000000; color: #FFFFFF;}
    .stFileUploader {background-color: #111111; border: 1px solid #333; border-radius: 5px;}
    h1, h2, h3 {color: #00FFCC !important;}
    div[data-testid="stMetricValue"] {color: #00FFCC !important;}
    p, label, li {color: #E0E0E0 !important;}
    .stProgress > div > div > div > div {background-color: #00FFCC;}
    button {border: 1px solid #00FFCC !important; color: #00FFCC !important;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.keras')

def process_image(image):
    image = ImageOps.fit(image, (128, 128), Image.LANCZOS)
    img_array = np.asarray(image).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name='top_activation', pred_index=None):
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        if isinstance(preds, list):
            preds = preds[0]
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
            
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    if isinstance(last_conv_layer_output, list):
        last_conv_layer_output = last_conv_layer_output[0]

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def create_pdf(image, result, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 10, "ALZHEIMER DIAGNOSIS REPORT", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pdf.cell(0, 10, f"Date: {current_time}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, f"Diagnosis Result: {result}", ln=True)
    pdf.cell(0, 10, f"Confidence: {confidence:.2f}%", ln=True)
    pdf.ln(10)
    
    tmp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            image.save(tmp_file.name)
            tmp_file_path = tmp_file.name
        
        pdf.image(tmp_file_path, x=10, y=80, w=100)
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        
    return pdf.output(dest="S").encode("latin-1")

try:
    model = load_model()
except:
    st.error("Model file not found. Please check my_alzheimer_model.keras")
    st.stop()

CLASS_NAMES = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

st.title("Alzheimer Prediction System")

file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if file is not None:
    original_image = Image.open(file).convert('RGB')
    processed_img = process_image(original_image)
    
    predictions = model.predict(processed_img)
    scores = tf.nn.softmax(predictions[0]).numpy()
    
    final_class = CLASS_NAMES[np.argmax(scores)]
    confidence = np.max(scores) * 100

    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        st.subheader("Diagnostic Result")
        color_code = "#00FF00" if final_class == "Non Demented" else "#FF3333"
        st.markdown(f"<h2 style='color: {color_code} !important;'>{final_class}</h2>", unsafe_allow_html=True)
        st.metric("Confidence", f"{confidence:.2f}%")
        
        st.write("---")
        st.write("Probability Distribution:")
        for i, name in enumerate(CLASS_NAMES):
            val = int(scores[i] * 100)
            st.write(f"{name}")
            st.progress(val)
            st.caption(f"{val}%")
            
        pdf_bytes = create_pdf(original_image, final_class, confidence)
        st.download_button(label="Download Medical Report (PDF)", data=pdf_bytes, file_name="report.pdf", mime="application/pdf")

    with col2:
        st.subheader("Original Image")
        st.image(original_image, use_container_width=True)

    with col3:
        st.subheader("Heatmap")
        try:
            heatmap = make_gradcam_heatmap(processed_img, model)
            
            heatmap_resized = cv2.resize(heatmap, (original_image.size[0], original_image.size[1]))
            
            heatmap_resized = np.uint8(255 * heatmap_resized)
            heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            superimposed = cv2.addWeighted(np.array(original_image), 0.6, heatmap_colored, 0.4, 0)
            st.image(superimposed, caption="Red areas indicate disease features", use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate heatmap: {e}")