import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="🖼️ Image Editor", layout="wide")

# Custom styles for pastel background and header
st.markdown("""
    <style>
        .main {
            background-color: #f6f5f3;
        }
        header {
            background-color: #e0dede;
            padding: 20px;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background-color: #e0dede; padding: 20px; border-radius: 8px;'>
    <h1 style='text-align: center; color: #160148'>🎨 Image Editor Web App</h1>
    <p style='text-align: center; color: #160148''>Just Upload an image<br> And <br> Apply filters like grayscale, sepia, sharpen, smoothing, manual cropping, and more.</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Original Image", use_container_width=True)

    img_array = np.array(image)
    if img_array.shape[-1] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    st.markdown("---")

    # Manual Crop Section
    st.subheader("✂️ Crop Image")
    h, w = img_bgr.shape[:2]
    st.write(f"🖼️ Image dimensions: **{w}x{h}** (width x height)")
    col1, col2 = st.columns(2)
    with col1:
        x = st.slider("Start X", 0, w - 1, 0)
        crop_width = st.slider("Crop Width", 1, w - x, w // 2)
    with col2:
        y = st.slider("Start Y", 0, h - 1, 0)
        crop_height = st.slider("Crop Height", 1, h - y, h // 2)
    cropped_img = img_bgr[y:y+crop_height, x:x+crop_width]

    processed_img = cropped_img.copy()

    # Filters Dropdown
    st.subheader("🎨 Filters")
    filters = st.selectbox("Choose a filter", ["None", "Grayscale", "Sepia", "Invert Colors", "Pencil Sketch"])

    if filters == "Grayscale":
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    elif filters == "Sepia":
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        processed_img = cv2.transform(processed_img, sepia_filter)
        processed_img = np.clip(processed_img, 0, 255).astype(np.uint8)
    elif filters == "Invert Colors":
        processed_img = cv2.bitwise_not(processed_img)
    elif filters == "Pencil Sketch":
        gray, sketch = cv2.pencilSketch(processed_img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
        processed_img = sketch

    # Smoothing/Sharpening Dropdown
    st.subheader("🔧 Smoothing & Sharpening")
    smooth_option = st.selectbox("Choose smoothing or sharpening", ["None", "Sharpen", "Gaussian Blur"])

    if smooth_option == "Sharpen":
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        processed_img = cv2.filter2D(processed_img, -1, kernel)
    elif smooth_option == "Gaussian Blur":
        ksize = st.slider("Blur Kernel Size", 1, 25, 5, step=2)
        processed_img = cv2.GaussianBlur(processed_img, (ksize, ksize), 0)

    # Edge Detection
    st.subheader("🧠 Edge Detection")
    if st.checkbox("Apply Canny Edge Detection"):
        if len(processed_img.shape) == 3:
            gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed_img
        processed_img = cv2.Canny(gray, 100, 200)

    st.markdown("---")
    st.subheader("🖼️ Processed Image")
    if len(processed_img.shape) == 2:
        st.image(processed_img, caption="Final Output", use_container_width=True, channels="GRAY")
    else:
        img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Final Output", use_container_width=True)

    st.markdown("---")
    st.subheader("⬇️ Download Your Image")
    download_btn = st.button("📥 Download Result")
    if download_btn:
        if len(processed_img.shape) == 2:
            result = Image.fromarray(processed_img)
        else:
            result = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button("Click to Download", data=byte_im, file_name="edited_image.png", mime="image/png")
