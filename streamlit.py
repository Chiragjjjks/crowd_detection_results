import streamlit as st
import cv2
import time
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# Load YOLO model
model = YOLO('weights/best.pt')

# Set up the Streamlit interface
st.title("Real-Time People Detection")
st.write("Upload a video or image, and we'll detect the number of people in each frame.")

# Confidence threshold slider
confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

# File uploader (support both images and videos)
uploaded_file = st.file_uploader("Upload a file", type=["mp4", "avi", "mov", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Process video
    if uploaded_file.type in ["video/mp4", "video/avi", "video/mov"]:
        with st.spinner("Processing video... Please wait"):
            start_time = time.time()

            # Save uploaded video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(uploaded_file.getbuffer())
                temp_video_path = temp_video.name

            # Open video capture
            cap = cv2.VideoCapture(temp_video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Fallback FPS if unavailable
            frame_width, frame_height = 640, 640

            # Temporary processed video file
            processed_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
            out = cv2.VideoWriter(processed_video_path, fourcc, fps, (frame_width, frame_height))

            # Placeholder for video and people count
            video_placeholder = st.empty()  # Placeholder to display video
            people_count_placeholder = st.empty()  # Placeholder for the people count text

            # Process frames and display in sync
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize and process the frame
                frame_resized = cv2.resize(frame, (frame_width, frame_height))
                results = model(frame_resized)

                # Count the number of people detected in the frame
                people_count = 0
                for box in results[0].boxes:
                    if int(box.cls[0]) == 0 and box.conf[0] >= confidence_threshold:  # Class 0 is 'person'
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        people_count += 1

                # Write the processed frame to the output video
                out.write(frame_resized)

                # Update the people count in real-time
                people_count_placeholder.markdown(f"*People Detected: {people_count}*")

                # Convert frame to display in Streamlit video
                _, buffer = cv2.imencode(".jpg", frame_resized)
                frame_bytes = buffer.tobytes()

                # Show current frame
                video_placeholder.image(frame_bytes, channels="BGR", use_column_width=True)

                # Simulate video playback speed
                time.sleep(1 / fps)

            # Release resources
            cap.release()
            out.release()

            # Final clean-up of temporary files
            os.remove(temp_video_path)

            end_time = time.time()
            st.write(f"Total Processing Time: {end_time - start_time:.2f} seconds")

    # Process image
    elif uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
        with st.spinner("Processing image... Please wait"):
            start_time = time.time()

            # Read the image
            image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            # Resize and process the image
            image_resized = cv2.resize(image, (640, 640))
            results = model(image_resized)

            # Count the number of people detected in the image
            people_count = 0
            for box in results[0].boxes:
                if int(box.cls[0]) == 0 and box.conf[0] >= confidence_threshold:  # Class 0 is 'person'
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(image_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    people_count += 1

            # Show the processed image and people count
            st.image(image_resized, channels="BGR", use_column_width=True)
            st.markdown(f"*People Detected: {people_count}*")

            end_time = time.time()
            st.write(f"Total Processing Time: {end_time - start_time:.2f} seconds")