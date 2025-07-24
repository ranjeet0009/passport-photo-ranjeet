import streamlit as st
from PIL import Image, ImageOps
from rembg import remove
import io
import cv2
import numpy as np

st.set_page_config(page_title="Passport Photo Maker", layout="centered")

st.title("🪪 Passport Photo Maker with AI")
st.markdown("Upload a clear face photo. The app will remove the background and format it for passport/ID use.")

uploaded_file = st.file_uploader("📤 Upload a Photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image from uploaded file
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("📷 Original Image")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert PIL to bytes for rembg
    with st.spinner("🧠 Removing background using AI..."):
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()

        # Remove background
        output_bytes = remove(img_bytes)
        result_image = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

    # Convert transparent background to white
    white_bg = Image.new("RGBA", result_image.size, (255, 255, 255, 255))
    final_image = Image.alpha_composite(white_bg, result_image).convert("RGB")

    # Resize to standard passport size (e.g., 600x600 pixels)
    passport_size = (600, 600)
    final_image = ImageOps.fit(final_image, passport_size, Image.ANTIALIAS, centering=(0.5, 0.5))

    st.subheader("✅ Processed Passport Photo")
    st.image(final_image, caption="Background Removed", use_column_width=False)

    # Download button
    img_io = io.BytesIO()
    final_image.save(img_io, format='JPEG', quality=95)
    st.download_button(
        label="📥 Download Passport Photo",
        data=img_io.getvalue(),
        file_name="passport_photo.jpg",
        mime="image/jpeg"
    )

else:
    st.info("Upload a high-quality front-facing image for best results.")
