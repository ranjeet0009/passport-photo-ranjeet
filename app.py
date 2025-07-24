import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from rembg import remove
import io

# Constants
PASSPORT_SIZE = (413, 531)  # Standard 2x2 inch photo at 300 DPI

st.set_page_config(page_title="Passport Photo Generator", layout="centered")
st.title("🪪 Passport Size Photo Generator")

uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Load and preview
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Uploaded Image", use_column_width=True)

        # Remove background
        st.info("Processing image and removing background...")
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()
        output = remove(img_bytes)  # returns binary

        # Load result into Pillow
        cleaned = Image.open(io.BytesIO(output)).convert("RGBA")

        # Replace background with white
        bg = Image.new("RGBA", cleaned.size, (255, 255, 255, 255))
        final = Image.alpha_composite(bg, cleaned).convert("RGB")

        # Resize and center crop to passport size
        passport = ImageOps.fit(final, PASSPORT_SIZE, method=Image.LANCZOS, centering=(0.5, 0.5))

        # Show result
        st.success("✅ Passport photo generated successfully!")
        st.image(passport, caption="Passport Size Photo", use_column_width=False)

        # Download
        buffer = io.BytesIO()
        passport.save(buffer, format="JPEG")
        st.download_button("📥 Download Passport Photo", buffer.getvalue(), file_name="passport_photo.jpg", mime="image/jpeg")

    except Exception as e:
        st.error("⚠️ Something went wrong while processing the image.")
        st.exception(e)
        
