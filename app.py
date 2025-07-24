import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter  # Added ImageFilter here
import tempfile
from rembg import remove
import io
import os

# Constants
PASSPORT_SIZE = (413, 531)  # 35x45mm @ 300 DPI
FACE_HEIGHT_RATIO = 0.60  # Face occupies 60% of photo height
TOP_SPACE_RATIO = 0.20  # 20% space above head
SHOULDER_EXTENSION = 0.25  # 25% additional space below face for shoulders

# Load OpenCV's DNN face detection model
prototxt_path = "deploy.prototxt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"

# Download model files if they don't exist
if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
    import urllib.request
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        prototxt_path
    )
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        model_path
    )

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Title and instructions
st.title("Professional Passport Photo Generator")
st.markdown("""
Upload any photo to get a perfectly standardized passport photo with:
- Centered face
- Proper headroom
- Shoulders included
- White background
""")

def detect_hair_region(np_img, face_box):
    """Estimate hair region above detected face"""
    (x, y, w, h) = face_box
    # Extend above face by 30% of face height for hair
    hair_height = int(h * 0.3)
    hair_y1 = max(y - hair_height, 0)
    hair_box = (x, hair_y1, w, hair_height)
    return hair_box

def standardize_passport_photo(image):
    """Process image to standardized passport photo"""
    # Remove background
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_nobg = remove(img_byte_arr.getvalue())
    img_nobg = Image.open(io.BytesIO(img_nobg)).convert("RGBA")
    
    # Create white background
    white_bg = Image.new("RGBA", img_nobg.size, (255, 255, 255, 255))
    img_white = Image.alpha_composite(white_bg, img_nobg).convert("RGB")
    
    # Convert to numpy array for face detection
    np_img = np.array(img_white)
    (h, w) = np_img.shape[:2]
    
    # Detect faces using OpenCV DNN
    blob = cv2.dnn.blobFromImage(cv2.resize(np_img, (300, 300)), 1.0,
                                (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    # Find the face with highest confidence
    max_confidence = 0
    best_face = None
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > max_confidence:
            max_confidence = confidence
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            best_face = (startX, startY, endX - startX, endY - startY)
    
    if best_face is None:
        raise ValueError("No face detected with sufficient confidence")
    
    (x, y, w, h) = best_face
    
    # Estimate hair region
    hair_box = detect_hair_region(np_img, best_face)
    
    # Calculate required dimensions with adjustments
    face_height = h
    total_height = int(face_height / FACE_HEIGHT_RATIO)
    top_space = int(total_height * TOP_SPACE_RATIO)
    shoulder_space = int(face_height * SHOULDER_EXTENSION)
    
    # Calculate crop coordinates with hair protection
    y1 = max(y - top_space, 0, hair_box[1])  # Don't crop above detected hair
    y2 = min(y + h + shoulder_space, np_img.shape[0])
    
    # Calculate width maintaining aspect ratio
    target_aspect = PASSPORT_SIZE[0] / PASSPORT_SIZE[1]
    required_width = int((y2 - y1) * target_aspect)
    
    # Center horizontally on face
    face_center = x + w // 2
    x1 = max(face_center - required_width // 2, 0)
    x2 = min(x1 + required_width, np_img.shape[1])
    
    # Adjust if at image boundaries
    if x2 - x1 < required_width:
        x1 = max(x2 - required_width, 0)
    
    # Crop and resize
    cropped = np_img[y1:y2, x1:x2]
    passport_img = Image.fromarray(cropped)
    
    # Apply slight blur to edges for smoother transitions
    passport_img = passport_img.filter(ImageFilter.GaussianBlur(radius=0.7))
    
    passport_img = ImageOps.fit(passport_img, PASSPORT_SIZE, method=Image.Resampling.LANCZOS)
    
    # Final white background with better edge handling
    final_img = Image.new("RGB", PASSPORT_SIZE, (255, 255, 255))
    final_img.paste(passport_img, (0, 0), passport_img.convert("RGBA") if passport_img.mode == 'RGBA' else None)
    
    return final_img

# File uploader
uploaded_file = st.file_uploader("Upload photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        original_img = Image.open(uploaded_file).convert("RGB")
        
        with st.spinner('Generating professional passport photo...'):
            result_img = standardize_passport_photo(original_img)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_img, caption="Original Photo", use_column_width=True)
        with col2:
            st.image(result_img, caption="Professional Passport Photo", use_column_width=True)
        
        # Download button
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            result_img.save(tmp_file.name, quality=100, subsampling=0)
            st.download_button(
                "Download Passport Photo",
                data=open(tmp_file.name, "rb").read(),
                file_name="passport_photo.jpg",
                mime="image/jpeg"
            )
    
    except ValueError as e:
        st.error(f"Error: {str(e)}. Please upload a clearer front-facing photo.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Add styling information
st.sidebar.markdown("""
### Photo Requirements:
- Front-facing, clear view of face
- Neutral expression
- No hats or heavy headwear
- Plain background preferred
""")
