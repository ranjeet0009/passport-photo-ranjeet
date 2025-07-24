import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from rembg import remove
import face_recognition
import io

st.title("🎓 Passport Size Photo Generator")
st.markdown("Upload any photo — the app will detect the face, crop it, remove background, and output a **2x2 inch passport-size photo with white background** at 300 DPI.")

uploaded_file = st.file_uploader("📤 Upload your photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    # Step 1: Remove Background
    no_bg = remove(image)

    # Step 2: Face Detection using AI (face_recognition)
    image_np = np.array(no_bg)
    face_locations = face_recognition.face_locations(image_np)

    if face_locations:
        top, right, bottom, left = face_locations[0]
        face_image = no_bg.crop((left, top, right, bottom))

        # Step 3: Resize to 2x2 inch at 300 DPI (600x600 px)
        passport_size = (600, 600)
        face_image = ImageOps.fit(face_image, passport_size, method=Image.LANCZOS)

        # Step 4: Paste on white background
        final_img = Image.new("RGB", passport_size, (255, 255, 255))
        final_img.paste(face_image, (0, 0))

        st.image(final_img, caption="🪪 Final Passport Photo", use_column_width=False)

        # Download
        img_bytes = io.BytesIO()
        final_img.save(img_bytes, format="JPEG", dpi=(300, 300))
        st.download_button("📥 Download Passport Photo", data=img_bytes.getvalue(), file_name="passport_photo.jpg", mime="image/jpeg")

    else:
        st.error("😔 Could not detect face. Please upload a clear front-facing photo.")
