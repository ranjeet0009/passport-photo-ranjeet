import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import tempfile
from rembg import remove
import io

# Constants
PASSPORT_SIZE = (413, 531)  # 35x45mm @ 300 DPI
HEAD_TO_SHOULDER_RATIO = 0.75  # Head occupies 75% of total height
TOP_SPACE_RATIO = 0.1  # 10% space above head

# Title
st.title("Professional Passport Photo Maker")

# Instructions
st.sidebar.markdown("### Instructions:")
st.sidebar.markdown("1. Upload a clear front-facing photo")
st.sidebar.markdown("2. Ensure your face is clearly visible")
st.sidebar.markdown("3. The system will automatically adjust to passport standards")

uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read and process image
    with st.spinner('Creating your passport photo...'):
        # Load image
        image = Image.open(uploaded_file).convert("RGB")
        
        # Remove background
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        image_nobg = remove(img_byte_arr.getvalue())
        image_nobg = Image.open(io.BytesIO(image_nobg)).convert("RGBA")
        
        # Create white background
        white_bg = Image.new("RGBA", image_nobg.size, (255, 255, 255, 255))
        image_white_bg = Image.alpha_composite(white_bg, image_nobg).convert("RGB")

        # Convert to OpenCV for face detection
        cv_image = np.array(image_white_bg)[:, :, ::-1].copy()
        
        # Detect face with more accurate parameters
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            st.error("No face detected. Please upload a clear front-facing photo.")
            st.stop()
        
        # Get the largest face detected
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        
        # Calculate crop dimensions based on face size
        total_height = int(h / HEAD_TO_SHOULDER_RATIO)
        top_space = int(total_height * TOP_SPACE_RATIO)
        
        # Calculate crop coordinates
        y1 = max(y - top_space, 0)
        y2 = min(y1 + total_height, cv_image.shape[0])
        
        # Calculate width maintaining aspect ratio
        target_aspect = PASSPORT_SIZE[0] / PASSPORT_SIZE[1]
        required_width = int((y2 - y1) * target_aspect)
        
        # Center the crop horizontally
        face_center = x + w//2
        x1 = max(face_center - required_width//2, 0)
        x2 = min(x1 + required_width, cv_image.shape[1])
        
        # Adjust if we're at image boundaries
        if x2 - x1 < required_width:
            x1 = max(x2 - required_width, 0)
        
        # Final crop
        cropped = cv_image[y1:y2, x1:x2]
        
        # Convert to PIL and resize to passport size
        passport_photo = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        passport_photo = ImageOps.fit(passport_photo, PASSPORT_SIZE, method=Image.Resampling.LANCZOS)
        
        # Ensure pure white background
        final_image = Image.new("RGB", PASSPORT_SIZE, (255, 255, 255))
        final_image.paste(passport_photo, (0, 0))

    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Photo", use_column_width=True)
    with col2:
        st.image(final_image, caption="Passport Photo", use_column_width=True)
    
    # Download button
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        final_image.save(tmp_file.name, quality=100, subsampling=0)
        st.download_button(
            "Download Passport Photo",
            data=open(tmp_file.name, "rb").read(),
            file_name="passport_photo.jpg",
            mime="image/jpeg"
        )
