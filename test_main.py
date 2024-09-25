from ultralytics import YOLO
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained violence detection model
violence_model = load_model('model_v_nv/model.h5')

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")

# Define input size for MobileNet
input_size = (128, 128)  # Adjust based on your model's input size
violence_threshold = 30  # Set threshold for 'Violent' classification

# Define colors
person_color = (255, 0, 0)  # Light blue color for person bounding box
border_thickness = 1  # Thin border thickness

# Input video file
video_path = 'assault_trim.mp4'  # Path to the input video file
output_path = 'prototype.mp4'  # Path to save the output video

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# YOLO class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

cv2.namedWindow('Video Feed', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Video Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break  # Exit if the video has ended

    # YOLO detection
    results = yolo_model(img, stream=True)

    # Process YOLO results
    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])
            
            if classNames[cls] == "person":
                # Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

                # Draw translucent blue bounding box
                overlay = img.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), -1)  # Fill the rectangle with blue
                cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)  # Apply transparency
                
                # Draw thin border
                cv2.rectangle(img, (x1, y1), (x2, y2), person_color, border_thickness)

    # Preprocess the frame for violence detection
    img_resized = cv2.resize(img, input_size)
    img_normalized = img_resized / 255.0  # Normalize if your model was trained with normalized inputs
    img_expanded = np.expand_dims(img_normalized, axis=0)

    # Make prediction
    prediction = violence_model.predict(img_expanded)
    violent_confidence = prediction[0][0] * 100  # Confidence for 'Violent' class

    # Apply threshold logic
    if violent_confidence < violence_threshold:
        label = 'Non-Violent'
        confidence = 100 - violent_confidence  # Reflect confidence for 'Non-Violent'
    else:
        label = 'Violent'
        confidence = violent_confidence  # Reflect confidence for 'Violent'

    # Display the violence detection result
    # Define the text
    display_text = "CAM 1"

    # Get the text size
    (text_width, text_height), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

    # Define the position for the text
    text_x, text_y = 10, 30

    # Draw a black rectangle as the background for the text
    cv2.rectangle(img, (text_x, text_y - text_height - baseline), (text_x + text_width, text_y + baseline), (0, 0, 0), thickness=cv2.FILLED)

    # Put the white text on the black background
    cv2.putText(img, display_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    if label == 'Violent':
        height, width, _ = img.shape
        text = "Violence Detected!"
        position = (width // 2 - 150, height // 2)  # Center of the image
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Adjusted values # Bright red color
        
        alert_position = (width // 2 - 150, height // 2 + 40)  # 40 pixels below the previous text
        cv2.putText(img, "Alert Raised", alert_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Green text for "Alert Raised"

    # Write the frame to the output video
    out.write(img)

    # Show the frame in full screen
    cv2.imshow('Video Feed', img)

    # Break the loop on 'Enter' key press
    if cv2.waitKey(10) == 13:  # Press Enter to exit
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
