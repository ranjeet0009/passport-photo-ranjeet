import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import mediapipe as mp
from rembg import remove
import io

st.set_page_config(page_title="Passport Photo Maker", layout="centered")

st.title("🪪 Passport Size Photo Generator (AI-Based)")

uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Face detection using mediapipe
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image_np)

    if not results.detections:
        st.error("No face detected. Please upload a photo with a clear front face.")
    else:
        # Get face bounding box
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box

        ih, iw, _ = image_np.shape
        x = int(bboxC.xmin * iw) - 30
        y = int(bboxC.ymin * ih) - 50
        w = int(bboxC.width * iw) + 60
        h = int(bboxC.height * ih) + 100

        x, y = max(x, 0), max(y, 0)

        # Crop the face region
        face_img = image.crop((x, y, x + w, y + h))

        # Remove background
        face_bytes = io.BytesIO()
        face_img.save(face_bytes, format='PNG')
        output = remove(face_bytes.getvalue())

        # Convert back to Image
        clean_img = Image.open(io.BytesIO(output)).convert("RGBA")

        # Add white background
        white_bg = Image.new("RGBA", clean_img.size, (255, 255, 255, 255))
        final_img = Image.alpha_composite(white_bg, clean_img).convert("RGB")

        # Resize to passport size (2x2 inches at 300 DPI = ~600x600 or 413x531 pixels)
        passport_size = (413, 531)
        final_passport = ImageOps.fit(final_img, passport_size, Image.LANCZOS)

        st.success("✅ Passport photo generated successfully!")
        st.image(final_passport, caption="Passport Size Photo", use_column_width=False)

        # Download
        img_io = io.BytesIO()
        final_passport.save(img_io, format='JPEG')
        st.download_button("📥 Download Passport Photo", data=img_io.getvalue(), file_name="passport_photo.jpg", mime="image/jpeg")
