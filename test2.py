
#A simpler version of the code which takes less ram to run and deploy


from ultralytics import YOLO
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained violence detection model
violence_model = load_model('model_v_nv/model.h5')

# Set up video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")

# Define input size for MobileNet
input_size = (128, 128)  # Adjust based on your model's input size
non_violent_threshold = 60  # Set threshold for classification

# Define colors
person_color = (255, 0, 0)  # Light blue color for person bounding box
border_thickness = 1  # Thin border thickness

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success, img = cap.read()
    
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
    class_label = np.argmax(prediction, axis=1)  # Get the predicted class
    confidence = np.max(prediction) * 100  # Get the confidence level as a percentage

    # Determine label and threshold
    labels = ['Non-Violent', 'Violent']  # Modify according to your dataset
    label = labels[class_label[0]]

    # Apply threshold logic
    if label == 'Non-Violent' and confidence > non_violent_threshold:
        label = 'Violent'
        confidence = 100 - confidence

    # Display the violence detection result
    if label == 'Violent':
        height, width, _ = img.shape
        text = "Violence Detected!"
        position = (width // 2 - 150, height // 2)  # Center of the image
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)  # Bright red color

    # Show the frame
    cv2.imshow('Live Video Feed', img)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
