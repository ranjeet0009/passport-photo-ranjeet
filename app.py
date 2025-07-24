import streamlit as st
import cv2
import numpy as np
from rembg import remove
from PIL import Image

st.title("Passport Size Photo Cropper and Background Remover")

uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Convert BGR to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption='Original Image', use_column_width=True)

    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        st.warning("No face detected.")
    else:
        for (x, y, w, h) in faces:
            padding = 40
            x1 = max(x - padding, 0)
            y1 = max(y - padding, 0)
            x2 = min(x + w + padding, img.shape[1])
            y2 = min(y + h + padding, img.shape[0])

            face_crop = img[y1:y2, x1:x2]

            # Convert to PIL for rembg
            face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

            # Remove background
            no_bg = remove(face_pil)

            # Replace background with white
            output = Image.new("RGBA", no_bg.size, (255, 255, 255, 255))
            output.paste(no_bg, mask=no_bg.split()[3])

            # Resize to passport size: 413x531 px (approx 35x45 mm at 300dpi)
            output = output.resize((413, 531))

            st.image(output, caption="Final Passport Photo", use_column_width=False)

            # Download button
            st.download_button("Download Passport Photo", data=output.tobytes(), file_name="passport_photo.png", mime="image/png")
