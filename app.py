import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64

# -------------------------
# Background Image
# -------------------------
def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        color: white; /* ✅ All text white */
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("assets/background.png")

# -------------------------
# Load Model
# -------------------------
model = tf.keras.models.load_model("model.h5")

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
img_size = 224

def prediction(img):
    img = img.convert("L")   # Grayscale
    img = img.resize((img_size, img_size))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    return predicted_class, confidence

# -------------------------
# Streamlit UI
# -------------------------
st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload MRI image and the model will predict the tumor type.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded MRI Image", use_container_width=True)

    predicted_class, confidence = prediction(img)
    result = classes[predicted_class]

    # -------------------------
    # Stylish Result Card
    # -------------------------
    if result == "notumor":
        emoji = "🟢"
        color = "lime"
    else:
        emoji = "⚠️"
        color = "red"

    st.markdown(
        f"""
        <div style="
            padding: 25px;
            border-radius: 15px;
            background-color: rgba(0,0,0,0.6);
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
            margin-top: 25px;
            color: white;
        ">
            {emoji} <span style="color:{color};">{result.upper()}</span>  
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Confidence:")
    st.progress(int(confidence))  # progress bar
    st.write(f"**{confidence:.2f}%**")
