import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import tempfile
from rembg import remove
import io
import os
import base64 # Import base64 module

# ============================================
# Custom CSS for Beautiful UI (Dark Theme)
# ============================================
st.markdown("""
<style>
    /* Main container */
    .stApp {
        background-color: #1a1a1a; /* Dark background */
        color: #e0e0e0; /* Light text color */
    }
    
    /* Header */
    .header {
        color: #e0e0e0; /* Light header text */
        font-size: 2.5em;
        text-align: center;
        margin-bottom: 0.5em;
    }
    
    /* File uploader */
    .stFileUploader > div {
        border: 2px dashed #4a90e2;
        border-radius: 10px;
        padding: 2em;
        background-color: #2c2c2c; /* Darker background for uploader */
        color: #e0e0e0; /* Light text */
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #4a90e2;
        color: black; /* Changed button font color to black */
        border-radius: 8px;
        padding: 0.5em 1em;
        font-weight: bold;
        border: none;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #357abd;
        transform: scale(1.02);
    }
    
    /* Cards for before/after */
    .card {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3); /* Darker shadow for dark theme */
        padding: 1em;
        background-color: #2c2c2c; /* Darker background for cards */
        color: #e0e0e0; /* Light text */
    }
    
    /* Progress spinner */
    .stSpinner>div {
        color: #4a90e2;
    }
    
    /* Sidebar - Targeting Streamlit's internal data-testid for better control */
    [data-testid="stSidebar"] {
        background-color: #2c2c2c; /* Changed sidebar background color */
    }
    [data-testid="stSidebarContent"] {
        background-color: #2c2c2c; /* Ensure content area also gets the background */
        color: white; /* Ensure text inside sidebar is white */
    }
    /* Specific styling for elements within the sidebar if needed */
    [data-testid="stSidebarContent"] h2, 
    [data-testid="stSidebarContent"] ol, 
    [data-testid="stSidebarContent"] ul, 
    [data-testid="stSidebarContent"] li,
    [data-testid="stSidebarContent"] small {
        color: white !important; /* Force white text for all sidebar elements */
    }
    
    /* Error messages */
    .stAlert {
        border-radius: 8px;
        background-color: #4a1c1c; /* Dark red for errors */
        color: #ffcccc; /* Light red text */
    }

    /* Adjust text colors within the main content */
    p {
        color: #e0e0e0; /* Default paragraph text color */
    }
    h3 {
        color: #e0e0e0; /* Default heading 3 text color */
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# App Configuration
# ============================================
st.set_page_config(
    page_title="Passport Photo Pro",
    page_icon="📸",
    layout="centered"
)

# Constants
PASSPORT_SIZE = (413, 531)  # 35x45mm @ 300 DPI
FACE_HEIGHT_RATIO = 0.50
TOP_SPACE_RATIO = 0.25 # Adjusted top space ratio for more space above head
SHOULDER_EXTENSION = 0.50 # Increased shoulder extension
ZOOM_OUT_FACTOR = 1.50 # Increased zoom out factor

# Helper functions to convert images to base64 (moved to top)
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def original_img_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# ============================================
# Beautiful Header
# ============================================
st.markdown("""
<div class="header">
    <span style="color: #4a90e2">📸 Passport Photo Pro</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<p style="text-align: center; color: #b0b0b0; font-size: 1.1em;">
    Create perfect passport photos in seconds with AI technology
</p>
""", unsafe_allow_html=True)

# ============================================
# Sidebar with Instructions
# ============================================
with st.sidebar:
    st.markdown("""
    <h2 style="color: white;">📋 Instructions</h2>
    <ol style="color: white;">
        <li>Upload a clear front-facing photo</li>
        <li>Ensure your face is visible</li>
        <li>Download your perfect passport photo</li>
    </ol>
    
    <h2 style="color: white;">✨ Tips</h2>
    <ul style="color: white;">
        <li>Use a plain background if possible</li>
        <li>Face the camera directly</li>
        <li>Keep a neutral expression</li>
    </ul>
    
    <div style="margin-top: 2em; text-align: center;">
        <small style="color: #bdc3c7;">Your photos are processed securely and never stored.</small>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# Main Content
# ============================================
# File uploader with custom styling
uploaded_file = st.file_uploader(
    "Drag & drop your photo here or click to browse",
    type=["jpg", "jpeg", "png"],
    help="We support JPG, JPEG, and PNG formats"
)

# Load face detection model
@st.cache_resource
def load_face_detection_model():
    prototxt_path = "deploy.prototxt"
    model_path = "res10_300x300_ssd_iter_140000.caffemodel"
    
    if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
        import urllib.request
        # Download prototxt
        try:
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                prototxt_path
            )
        except Exception as e:
            st.error(f"Error downloading deploy.prototxt: {e}")
            return None
        
        # Download model
        try:
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                model_path
            )
        except Exception as e:
            st.error(f"Error downloading caffemodel: {e}")
            return None
            
    try:
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        return net
    except Exception as e:
        st.error(f"Error loading face detection model: {e}")
        return None

net = load_face_detection_model()

# ============================================
# Photo Processing Functions
# ============================================
def detect_hair_region(np_img, face_box):
    (x, y, w, h) = face_box
    hair_height = int(h * 0.5)
    hair_y1 = max(y - hair_height, 0)
    return (x, hair_y1, w, hair_height)

def standardize_passport_photo(image):
    # Convert to numpy array for initial face detection
    np_img_original = np.array(image.convert("RGB"))
    (h_orig, w_orig) = np_img_original.shape[:2]
    
    # Check if face detection model is loaded
    if net is None:
        raise RuntimeError("Face detection model failed to load.")

    # Detect faces on the original image
    blob = cv2.dnn.blobFromImage(cv2.resize(np_img_original, (300, 300)), 1.0,
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
            box = detections[0, 0, i, 3:7] * np.array([w_orig, h_orig, w_orig, h_orig])
            (startX, startY, endX, endY) = box.astype("int")
            best_face = (startX, startY, endX - startX, endY - startY)
    
    if best_face is None:
        raise ValueError("No face detected with sufficient confidence")
    
    (x, y, w, h) = best_face
    
    # Estimate hair region
    hair_box = detect_hair_region(np_img_original, best_face)
    
    # Calculate dimensions for the initial crop with increased zoom-out and shoulder extension
    total_height = int((h / FACE_HEIGHT_RATIO) * ZOOM_OUT_FACTOR)
    top_space = int(total_height * TOP_SPACE_RATIO)
    shoulder_space = int(h * SHOULDER_EXTENSION) # Use increased shoulder extension
    
    # Calculate crop coordinates to isolate the person
    y1 = max(y - top_space, 0, hair_box[1])
    y2 = min(y + h + shoulder_space, np_img_original.shape[0])
    
    target_aspect = PASSPORT_SIZE[0] / PASSPORT_SIZE[1]
    required_width = int((y2 - y1) * target_aspect) # Do not apply ZOOM_OUT_FACTOR here again
    
    face_center = x + w // 2
    x1 = max(face_center - required_width // 2, 0)
    x2 = min(x1 + required_width, np_img_original.shape[1])
    
    # Ensure crop dimensions are valid
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Calculated crop dimensions are invalid. Please try another photo.")

    # Perform initial crop to get the region of interest (the person)
    cropped_for_rembg_np = np_img_original[y1:y2, x1:x2]
    cropped_for_rembg_pil = Image.fromarray(cropped_for_rembg_np)

    # Now, remove background from this cropped image
    img_byte_arr_cropped = io.BytesIO()
    cropped_for_rembg_pil.save(img_byte_arr_cropped, format='PNG')
    img_nobg_cropped = remove(img_byte_arr_cropped.getvalue())
    img_nobg_cropped_pil = Image.open(io.BytesIO(img_nobg_cropped)).convert("RGBA")
    
    # Create white background for the processed image
    white_bg_final = Image.new("RGBA", img_nobg_cropped_pil.size, (255, 255, 255, 255))
    img_white_final = Image.alpha_composite(white_bg_final, img_nobg_cropped_pil).convert("RGB")
    
    # Edge smoothing
    img_white_final = img_white_final.filter(ImageFilter.GaussianBlur(radius=1.0)) # Increased blur radius
    
    # Resize to passport size
    passport_img = ImageOps.fit(img_white_final, PASSPORT_SIZE, method=Image.Resampling.LANCZOS)
    
    # Final composition (this step is mostly for consistency, as img_white_final is already RGB with white bg)
    final_img = Image.new("RGB", PASSPORT_SIZE, (255, 255, 255))
    final_img.paste(passport_img, (0, 0))
    
    return final_img

# ============================================
# Main Processing Logic
# ============================================
if uploaded_file:
    if net is None: # Check if model loaded successfully before proceeding
        st.error("Cannot process photo: Face detection model failed to load. Please check your internet connection and try again.")
    else:
        try:
            original_img = Image.open(uploaded_file).convert("RGB")
            
            with st.spinner('✨ Creating your perfect passport photo...'):
                result_img = standardize_passport_photo(original_img)
            
            # Display results in cards
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; margin: 2em 0;">
                <div class="card" style="width: 48%;">
                    <h3 style="color: #e0e0e0; text-align: center;">Original Photo</h3>
                    <img src="data:image/png;base64,{original_img_to_base64(original_img)}" style="width: 100%; border-radius: 8px;">
                </div>
                <div class="card" style="width: 48%;">
                    <h3 style="color: #e0e0e0; text-align: center;">Passport Photo (25% Zoom Out)</h3>
                    <img src="data:image/png;base64,{image_to_base64(result_img)}" style="width: 100%; border-radius: 8px;">
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Download button
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                result_img.save(tmp_file.name, quality=100, subsampling=0)
                st.download_button(
                    "⬇️ Download Passport Photo",
                    data=open(tmp_file.name, "rb").read(),
                    file_name="passport_photo.jpg",
                    mime="image/jpeg",
                    help="Click to download your perfectly formatted passport photo"
                )
            os.unlink(tmp_file.name) # Clean up the temporary file
        
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}. Please try another photo or ensure your photo has a clear, front-facing face.")

# ============================================
# Footer
# ============================================
st.markdown("""
<div style="text-align: center; margin-top: 3em; color: #7f8c8d; font-size: 0.9em;">
    <hr style="border: 0.5px solid #3c3c3c;">
    <p>Passport Photo Pro - Create perfect passport photos in seconds</p>
    <p>© 2023 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
