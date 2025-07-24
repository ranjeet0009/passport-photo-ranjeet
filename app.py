import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import tempfile
from rembg import remove
import io

# Title
st.title("AI-Powered Passport Photo Maker")

# Sidebar with instructions
st.sidebar.markdown("### Instructions:")
st.sidebar.markdown("1. Upload a clear front-facing photo")
st.sidebar.markdown("2. Ensure your face is clearly visible")
st.sidebar.markdown("3. The system will automatically crop to passport size")

uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read the image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Remove background using AI
    with st.spinner('Processing image...'):
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
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))

    if len(faces) == 0:
        st.error("No face detected. Please upload a clear front-facing photo.")
    else:
        for (x, y, w, h) in faces:
            # Calculate crop area based on passport photo requirements
            # Standard passport photo shows head and top of shoulders
            head_height = h
            shoulders_height = int(h * 0.5)  # Additional space for shoulders
            
            # Calculate total height needed (head + some shoulders)
            total_height = head_height + shoulders_height
            
            # Calculate y-coordinate (start from top of head)
            y1 = max(y - int(h * 0.2), 0)  # Small space above head
            y2 = min(y1 + total_height, open_cv_image.shape[0])
            
            # Calculate width (maintain 35x45mm aspect ratio)
            target_ratio = 35/45  # width/height
            required_width = int((y2 - y1) * target_ratio)
            
            # Calculate x-coordinate (center the face)
            face_center_x = x + w//2
            x1 = max(face_center_x - required_width//2, 0)
            x2 = min(x1 + required_width, open_cv_image.shape[1])
            
            # Adjust if we hit image boundaries
            if x2 == open_cv_image.shape[1]:
                x1 = max(open_cv_image.shape[1] - required_width, 0)
            if y2 == open_cv_image.shape[0]:
                y1 = max(open_cv_image.shape[0] - total_height, 0)

            # Final crop
            face_img = open_cv_image[y1:y2, x1:x2]

            # Convert to PIL Image
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

            # Resize to passport photo size (35x45 mm @ 300 DPI = 413x531 pixels)
            passport_size = (413, 531)
            final_img = ImageOps.fit(face_pil, passport_size, method=Image.Resampling.LANCZOS)
            
            # Apply slight smoothing
            final_img = final_img.filter(ImageFilter.SMOOTH)

            # Create final image with pure white background
            result = Image.new("RGB", passport_size, (255, 255, 255))
            result.paste(final_img, (0, 0))

            # Display before/after
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            with col2:
                st.image(result, caption="Passport Photo", use_column_width=True)
            
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
