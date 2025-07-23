import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from rembg import remove
import face_recognition
import io

st.set_page_config(page_title="ID Card Photo Maker", layout="centered")
st.title("🧠 AI-Powered ID Card Photo Generator")
st.write("Upload your photo to auto-remove background, crop the face, and download the clean image.")

uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Step 1: Load image
    img = Image.open(uploaded_file).convert("RGB")

    # Step 2: Remove background
    with st.spinner("Removing background..."):
        no_bg = remove(img)

    # Step 3: Face detection for auto-cropping
    with st.spinner("Detecting and cropping face..."):
        img_array = np.array(no_bg)
        face_locations = face_recognition.face_locations(img_array)

        if face_locations:
            top, right, bottom, left = face_locations[0]
            cropped_face = img.crop((left - 50, top - 50, right + 50, bottom + 50))
        else:
            st.warning("Face not detected properly, showing original background-removed image.")
            cropped_face = no_bg

    # Step 4: Set white background
    with st.spinner("Finalizing image..."):
        final_img = Image.new("RGB", cropped_face.size, (255, 255, 255))
        final_img.paste(cropped_face, mask=cropped_face.split()[3] if cropped_face.mode == 'RGBA' else None)

        # Optional: Resize to fixed dimension (e.g. 600x800 for ID)
        output_img = final_img.resize((600, 800))

    st.image(output_img, caption="Processed ID Photo", use_column_width=False)

    # Step 5: Download button
    buf = io.BytesIO()
    output_img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    st.download_button("📥 Download Processed Image", data=byte_im, file_name="id_photo.jpg", mime="image/jpeg")
else:
    st.info("Upload a photo to begin.")