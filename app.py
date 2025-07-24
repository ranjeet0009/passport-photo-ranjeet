import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
from rembg import remove
import io

st.title("🎓 Passport Photo Generator with AI")
st.markdown("Upload your photo (any size/background). We'll detect the face, crop it, remove background, and output a white-background passport-sized photo (2x2 inch).")

uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Load OpenCV face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    if len(faces) == 0:
        st.error("❌ No face detected. Please upload a clearer image.")
    else:
        # Take first detected face
        (x, y, w, h) = faces[0]

        # Expand the face region a bit for passport framing
        padding = int(h * 1.5)
        cx, cy = x + w//2, y + h//2
        crop_x1 = max(cx - padding, 0)
        crop_y1 = max(cy - padding, 0)
        crop_x2 = min(cx + padding, image_np.shape[1])
        crop_y2 = min(cy + padding, image_np.shape[0])
        cropped = image_np[crop_y1:crop_y2, crop_x1:crop_x2]

        # Convert back to PIL
        cropped_img = Image.fromarray(cropped)

        # Remove background with rembg
        removed_bg = remove(cropped_img)

        # Convert transparent image to white background
        bg = Image.new("RGB", removed_bg.size, (255, 255, 255))
        bg.paste(removed_bg, mask=removed_bg.split()[3])  # Paste using alpha channel

        # Resize to passport size (2x2 inches at 300 DPI)
        passport_size = (600, 600)  # pixels
        final_img = ImageOps.fit(bg, passport_size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

        st.image(final_img, caption="🖼️ Final Passport Photo", use_column_width=False)

        # Download
        img_bytes = io.BytesIO()
        final_img.save(img_bytes, format="JPEG")
        st.download_button("📥 Download Passport Photo", data=img_bytes.getvalue(), file_name="passport_photo.jpg", mime="image/jpeg")
