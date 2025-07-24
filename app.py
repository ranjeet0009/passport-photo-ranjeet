import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
from rembg import remove
import io
import face_recognition

st.set_page_config(page_title="Passport Photo Maker", layout="centered")

st.title("🪪 AI Passport Photo Creator")
st.write("Upload any photo, and get a clean passport photo with white background!")

uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    with st.spinner("Processing..."):

        # Convert image to numpy array
        image_np = np.array(image)

        # Use face_recognition to locate faces
        face_locations = face_recognition.face_locations(image_np)

        if not face_locations:
            st.error("No face detected. Please upload a clearer image with a visible face.")
        else:
            # Take first face found
            top, right, bottom, left = face_locations[0]

            # Expand the box slightly for shoulders and top head
            height = bottom - top
            width = right - left
            expand_y = int(height * 0.5)
            expand_x = int(width * 0.3)

            top = max(top - expand_y, 0)
            bottom = min(bottom + expand_y, image_np.shape[0])
            left = max(left - expand_x, 0)
            right = min(right + expand_x, image_np.shape[1])

            face_crop = image_np[top:bottom, left:right]

            # Convert to PIL
            face_pil = Image.fromarray(face_crop)

            # Remove background
            buffered = io.BytesIO()
            face_pil.save(buffered, format="PNG")
            no_bg_data = remove(buffered.getvalue())
            no_bg_image = Image.open(io.BytesIO(no_bg_data)).convert("RGBA")

            # Replace transparent background with white
            white_bg = Image.new("RGBA", no_bg_image.size, (255, 255, 255, 255))
            final_image = Image.alpha_composite(white_bg, no_bg_image).convert("RGB")

            # Resize to passport size (approx. 413x531 pixels)
            passport_size = (413, 531)  # 3.5 x 4.5 cm at 300 DPI
            final_image = ImageOps.fit(final_image, passport_size, method=Image.Resampling.LANCZOS)

            st.success("Passport photo generated successfully!")
            st.image(final_image, caption="Passport Size Photo", use_column_width=False)

            # Provide download link
            buf = io.BytesIO()
            final_image.save(buf, format="JPEG")
            byte_im = buf.getvalue()
            st.download_button("📥 Download Passport Photo", data=byte_im, file_name="passport_photo.jpg", mime="image/jpeg")
