import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import tempfile
from rembg import remove
import io
import dlib  # More accurate face detection

# Constants
PASSPORT_SIZE = (413, 531)  # 35x45mm @ 300 DPI
FACE_HEIGHT_RATIO = 0.65  # Face should occupy 65% of photo height
TOP_SPACE_RATIO = 0.15  # 15% space above head

# Initialize dlib's face detector (more accurate than Haar cascades)
detector = dlib.get_frontal_face_detector()

# Title and instructions
st.title("AI Passport Photo Generator")
st.markdown("""
Upload any photo and get a perfectly standardized passport photo.
The AI will automatically center your face and adjust the zoom level.
""")

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
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    
    # Detect faces using dlib (more accurate)
    faces = detector(gray, 1)
    if len(faces) == 0:
        raise ValueError("No face detected")
    
    # Get the largest face
    face = max(faces, key=lambda f: f.width() * f.height())
    
    # Calculate required dimensions
    face_height = face.height()
    total_height = int(face_height / FACE_HEIGHT_RATIO)
    top_space = int(total_height * TOP_SPACE_RATIO)
    
    # Calculate crop coordinates
    y1 = max(face.top() - top_space, 0)
    y2 = min(y1 + total_height, np_img.shape[0])
    
    # Calculate width maintaining aspect ratio
    target_aspect = PASSPORT_SIZE[0] / PASSPORT_SIZE[1]
    required_width = int((y2 - y1) * target_aspect)
    
    # Center horizontally on face
    face_center = face.left() + face.width() // 2
    x1 = max(face_center - required_width // 2, 0)
    x2 = min(x1 + required_width, np_img.shape[1])
    
    # Adjust if at image boundaries
    if x2 - x1 < required_width:
        x1 = max(x2 - required_width, 0)
    
    # Crop and resize
    cropped = np_img[y1:y2, x1:x2]
    passport_img = Image.fromarray(cropped)
    passport_img = ImageOps.fit(passport_img, PASSPORT_SIZE, method=Image.Resampling.LANCZOS)
    
    # Final white background
    final_img = Image.new("RGB", PASSPORT_SIZE, (255, 255, 255))
    final_img.paste(passport_img, (0, 0))
    
    return final_img

# File uploader
uploaded_file = st.file_uploader("Upload photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        original_img = Image.open(uploaded_file).convert("RGB")
        
        with st.spinner('Generating standardized passport photo...'):
            result_img = standardize_passport_photo(original_img)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_img, caption="Original Photo", use_column_width=True)
        with col2:
            st.image(result_img, caption="Passport Photo", use_column_width=True)
        
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

# Add requirements information
st.sidebar.markdown("### Requirements:")
st.sidebar.markdown("- Face should be clearly visible")
st.sidebar.markdown("- No extreme angles")
st.sidebar.markdown("- Well-lit environment")
