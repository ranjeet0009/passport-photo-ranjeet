import streamlit as st
from PIL import Image, ImageOps
from rembg import remove
import io

st.set_page_config(page_title="Passport Photo Maker", layout="centered")

st.title("📸 Passport Photo Generator")

uploaded_file = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"])

if uploaded_file:
    with st.spinner("Processing..."):
        image = Image.open(uploaded_file).convert("RGBA")

        # Remove background
        output = remove(image)

        # Convert to white background
        white_bg = Image.new("RGBA", output.size, (255, 255, 255, 255))
        final_image = Image.alpha_composite(white_bg, output).convert("RGB")

        # Resize to passport photo size (e.g., 2x2 inches at 300 dpi = 600x600 pixels)
        passport_image = final_image.resize((600, 600))

        st.image(passport_image, caption="Passport Photo", use_column_width=False)

        # Download option
        img_bytes = io.BytesIO()
        passport_image.save(img_bytes, format="JPEG")
        st.download_button("Download Photo", data=img_bytes.getvalue(), file_name="passport_photo.jpg", mime="image/jpeg")
