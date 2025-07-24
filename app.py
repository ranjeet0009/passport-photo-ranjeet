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
FACE_HEIGHT_RATIO = 0.50 # Face height should be 50% of the total passport photo height
TOP_SPACE_RATIO = 0.15 # Top space (from top of photo to top of head) should be 15% of total height

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
    # This function is primarily for estimating the top of the head/hairline
    # (x, y, w, h) = face_box
    # hair_height = int(h * 0.5) # This might be too aggressive, let's simplify
    # hair_y1 = max(y - hair_height, 0)
    # return (x, hair_y1, w, hair_height)
    # For now, we'll rely on the TOP_SPACE_RATIO for top positioning
    return (0, 0, 0, 0) # Return dummy values as it's not directly used for crop calculation anymore

def standardize_passport_photo(image):
    # Step 1: Remove background from the original image
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_nobg = remove(img_byte_arr.getvalue())
    img_nobg_pil = Image.open(io.BytesIO(img_nobg)).convert("RGBA")
    
    # Create white background
    white_bg = Image.new("RGBA", img_nobg_pil.size, (255, 255, 255, 255))
    img_white = Image.alpha_composite(white_bg, img_nobg_pil).convert("RGB")
    
    # Convert to numpy array for face detection
    np_img_white = np.array(img_white)
    (h_white, w_white) = np_img_white.shape[:2]
    
    # Check if face detection model is loaded
    if net is None:
        raise RuntimeError("Face detection model failed to load.")

    # Detect faces on the image with white background
    blob = cv2.dnn.blobFromImage(cv2.resize(np_img_white, (300, 300)), 1.0,
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
            box = detections[0, 0, i, 3:7] * np.array([w_white, h_white, w_white, h_white])
            (startX, startY, endX, endY) = box.astype("int")
            best_face = (startX, startY, endX - startX, endY - startY)
    
    if best_face is None:
        raise ValueError("No face detected with sufficient confidence")
    
    (x, y, w, h) = best_face # Detected face bounding box on the original white-backgrounded image

    # Desired face height in the final PASSPORT_SIZE
    desired_face_height_in_passport = PASSPORT_SIZE[1] * FACE_HEIGHT_RATIO

    # Calculate scaling factor based on desired face height and detected face height
    scale_factor = desired_face_height_in_passport / h

    # Resize the entire image based on this scale factor
    scaled_width = int(w_white * scale_factor)
    scaled_height = int(h_white * scale_factor)
    scaled_img_pil = img_white.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
    
    # Now, recalculate face coordinates on the scaled image
    scaled_face_x = int(x * scale_factor)
    scaled_face_y = int(y * scale_factor)
    scaled_face_w = int(w * scale_factor)
    scaled_face_h = int(h * scale_factor) # This should be approximately desired_face_height_in_passport

    # Calculate the target position for the face within the final PASSPORT_SIZE
    # Top of the face (head) should be at TOP_SPACE_RATIO of the passport photo's height
    target_head_top_y_in_passport = int(PASSPORT_SIZE[1] * TOP_SPACE_RATIO)
    
    # Center the face horizontally in the passport photo
    target_face_center_x_in_passport = PASSPORT_SIZE[0] // 2
    
    # Calculate the crop box coordinates on the scaled image
    # The top of the crop should be such that the scaled_face_y aligns with target_head_top_y_in_passport
    crop_y1 = scaled_face_y - target_head_top_y_in_passport
    
    # The left of the crop should be such that the scaled_face_center_x aligns with target_face_center_x_in_passport
    crop_x1 = scaled_face_x + (scaled_face_w // 2) - target_face_center_x_in_passport

    # Calculate crop bottom and right based on PASSPORT_SIZE dimensions
    crop_y2 = crop_y1 + PASSPORT_SIZE[1]
    crop_x2 = crop_x1 + PASSPORT_SIZE[0]

    # Adjust crop box to stay within the bounds of the scaled image
    # Shift if left edge is negative
    if crop_x1 < 0:
        crop_x2 -= crop_x1 # Increase right bound by the negative amount
        crop_x1 = 0
    # Shift if top edge is negative
    if crop_y1 < 0:
        crop_y2 -= crop_y1 # Increase bottom bound by the negative amount
        crop_y1 = 0
    
    # Shift if right edge exceeds scaled image width
    if crop_x2 > scaled_width:
        crop_x1 -= (crop_x2 - scaled_width) # Decrease left bound
        crop_x2 = scaled_width
    # Shift if bottom edge exceeds scaled image height
    if crop_y2 > scaled_height:
        crop_y1 -= (crop_y2 - scaled_height) # Decrease top bound
        crop_y2 = scaled_height

    # Final check to ensure coordinates are not negative after adjustments
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    
    # Perform the final crop on the scaled image
    # Ensure crop dimensions are valid before cropping
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1 or \
       crop_x1 >= scaled_width or crop_y1 >= scaled_height:
        raise ValueError("Calculated final crop dimensions are invalid or out of bounds. Please try another photo.")

    final_cropped_pil = scaled_img_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    
    # Apply a light Gaussian blur for smoothing
    final_img_smoothed = final_cropped_pil.filter(ImageFilter.GaussianBlur(radius=0.5)) 
    
    # The image should already be PASSPORT_SIZE due to the precise cropping.
    # This final ImageOps.fit acts as a safeguard for exact dimensions and resampling quality.
    passport_img = ImageOps.fit(final_img_smoothed, PASSPORT_SIZE, method=Image.Resampling.LANCZOS)
    
    # Final composition (should already have white background)
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
            st.error(f"⚠️ Error: {str(e)}. Please try another photo or ensure your photo has a clear, front-facing face. Details: {e}")

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
