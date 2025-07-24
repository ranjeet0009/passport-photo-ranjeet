import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import tempfile
from rembg import remove
import io

# Title
st.title("AI-Powered Passport Photo Maker")

uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read the image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Remove background using AI
    st.write("Processing image...")
    
    # Convert to bytes for rembg
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Remove background
    image_nobg = remove(img_byte_arr)
    image_nobg = Image.open(io.BytesIO(image_nobg)).convert("RGBA")
    
    # Create white background
    white_bg = Image.new("RGBA", image_nobg.size, (255, 255, 255, 255))
    
    # Composite the image with white background
    image_white_bg = Image.alpha_composite(white_bg, image_nobg)
    image_white_bg = image_white_bg.convert("RGB")
    
    # Convert to OpenCV image for face detection
    open_cv_image = np.array(image_white_bg)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
    
    # Load Haarcascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.error("No face detected. Please upload a clear front-facing photo.")
    else:
        for (x, y, w, h) in faces:
            # Calculate margins (35x45mm aspect ratio)
            target_ratio = 45/35
            current_ratio = h/w
            
            if current_ratio < target_ratio:
                # Need more height
                new_h = int(w * target_ratio)
                margin = (new_h - h) // 2
                y1 = max(y - margin, 0)
                y2 = min(y + h + margin, open_cv_image.shape[0])
                x1 = max(x - int(0.2 * w), 0)
                x2 = min(x + w + int(0.2 * w), open_cv_image.shape[1])
            else:
                # Need more width
                new_w = int(h / target_ratio)
                margin = (new_w - w) // 2
                x1 = max(x - margin, 0)
                x2 = min(x + w + margin, open_cv_image.shape[1])
                y1 = max(y - int(0.3 * h), 0)
                y2 = min(y + h + int(0.3 * h), open_cv_image.shape[0])

            face_img = open_cv_image[y1:y2, x1:x2]

            # Convert to PIL Image
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

            # Resize to passport photo size (35x45 mm @ 300 DPI = 413x531 pixels)
            passport_size = (413, 531)
            final_img = ImageOps.fit(face_pil, passport_size, method=Image.Resampling.LANCZOS)
            
            # Apply slight smoothing to remove rough edges
            final_img = final_img.filter(ImageFilter.SMOOTH)

            # Create final image with pure white background
            result = Image.new("RGB", passport_size, (255, 255, 255))
            result.paste(final_img, (0, 0))

            st.image(result, caption="Passport Photo", use_column_width=False)
            st.success("Passport photo generated successfully!")

            # Download
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                result.save(tmp_file.name, quality=95, subsampling=0)
                st.download_button(
                    "Download Passport Photo", 
                    data=open(tmp_file.name, "rb").read(), 
                    file_name="passport_photo.jpg", 
                    mime="image/jpeg"
                )
            break
