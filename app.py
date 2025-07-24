import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import tempfile
from rembg import remove
import io
import os

# Constants
PASSPORT_SIZE = (413, 531)  # 35x45mm @ 300 DPI
FACE_HEIGHT_RATIO = 0.55  # Face occupies 55% of photo height
TOP_SPACE_RATIO = 0.25  # 25% space above head
SHOULDER_EXTENSION = 0.30  # 30% additional space below face for shoulders
ZOOM_OUT_FACTOR = 1.20  # 15% zoom out

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
- Full head coverage
- Proper shoulder inclusion
- White background
""")

def detect_hair_region(np_img, face_box):
    """Estimate hair region above detected face"""
    (x, y, w, h) = face_box
    hair_height = int(h * 0.4)  # 40% of face height for hair coverage
    hair_y1 = max(y - hair_height, 0)
    return (x, hair_y1, w, hair_height)

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
    
    # Calculate dimensions with zoom-out adjustments
    total_height = int((h / FACE_HEIGHT_RATIO) * ZOOM_OUT_FACTOR)
    top_space = int(total_height * TOP_SPACE_RATIO)
    shoulder_space = int(h * SHOULDER_EXTENSION)
    
    # Calculate crop coordinates with zoom-out
    y1 = max(y - top_space, 0, hair_box[1])
    y2 = min(y + h + shoulder_space, np_img.shape[0])
    
    # Calculate width with aspect ratio and zoom-out
    target_aspect = PASSPORT_SIZE[0] / PASSPORT_SIZE[1]
    required_width = int((y2 - y1) * target_aspect * ZOOM_OUT_FACTOR)
    
    # Center horizontally with zoom-out
    face_center = x + w // 2
    x1 = max(face_center - required_width // 2, 0)
    x2 = min(x1 + required_width, np_img.shape[1])
    
    # Final crop with zoom-out
    cropped = np_img[y1:y2, x1:x2]
    passport_img = Image.fromarray(cropped)
    
    # Gentle edge smoothing
    passport_img = passport_img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Resize to passport size
    passport_img = ImageOps.fit(passport_img, PASSPORT_SIZE, method=Image.Resampling.LANCZOS)
    
    # Final composition
    final_img = Image.new("RGB", PASSPORT_SIZE, (255, 255, 255))
    final_img.paste(passport_img, (0, 0))
    
    return final_img

# File uploader and processing
uploaded_file = st.file_uploader("Upload photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        original_img = Image.open(uploaded_file).convert("RGB")
        
        with st.spinner('Creating perfect passport photo...'):
            result_img = standardize_passport_photo(original_img)
        
        # Display comparison
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_img, caption="Original Photo", use_column_width=True)
        with col2:
            st.image(result_img, caption="Perfect Passport Photo", use_column_width=True)
        
        # Download
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            result_img.save(tmp_file.name, quality=100)
            st.download_button(
                "Download Photo",
                data=open(tmp_file.name, "rb").read(),
                file_name="passport_photo.jpg",
                mime="image/jpeg"
            )
    
    except Exception as e:
        st.error(f"Error: {str(e)}. Please try another photo.")

# Requirements
st.sidebar.markdown("""
### Perfect Passport Photos:
- Face centered naturally
- Full head and hair visible
- Professional shoulder framing
- Compliant with official standards
""")
