import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import tempfile
from rembg import remove

# Title
st.title("AI-Powered Passport Photo Maker")

uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Convert to PIL Image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Remove background using AI
    st.write("Removing background...")
    image_nobg = remove(image)
    
    # Convert to OpenCV image for face detection
    open_cv_image = np.array(image_nobg)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    
    # Convert transparent background to white
    open_cv_image[np.all(open_cv_image == [0, 0, 0, 0], axis=2)] = [255, 255, 255, 255]
    
    # Convert back to 3-channel image (remove alpha channel)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGRA2BGR)

    # Load Haarcascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.error("No face detected. Please upload a clear front-facing photo.")
    else:
        for (x, y, w, h) in faces:
            margin = int(0.6 * h)
            x1 = max(x - margin, 0)
            y1 = max(y - margin, 0)
            x2 = min(x + w + margin, open_cv_image.shape[1])
            y2 = min(y + h + margin, open_cv_image.shape[0])

            face_img = open_cv_image[y1:y2, x1:x2]

            # Convert to PIL Image
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

            # Resize to passport photo size (35x45 mm @ 300 DPI = 413x531 pixels)
            passport_size = (413, 531)
            final_img = ImageOps.fit(face_pil, passport_size, method=Image.Resampling.LANCZOS)

            # Create white background image
            white_bg = Image.new("RGB", passport_size, (255, 255, 255))
            white_bg.paste(final_img, (0, 0), final_img.convert("RGBA") if final_img.mode == 'RGBA' else None)

            st.image(white_bg, caption="Passport Photo", use_column_width=False)
            st.success("Passport photo generated successfully!")

            # Download
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                white_bg.save(tmp_file.name, quality=95)
                st.download_button(
                    "Download Passport Photo", 
                    data=open(tmp_file.name, "rb").read(), 
                    file_name="passport_photo.jpg", 
                    mime="image/jpeg"
                )
            break
