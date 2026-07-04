# pip install opencv-python
# pip install ultralytics
# https://roboflow.com/use-opencv/read-video-streams-with-cv2-videocapture

import cv2
from ultralytics import YOLO

# Load your trained YOLOv11 model
model = YOLO(r"C:\z_Learn\miraj_pandas\Deep Learning\YOLO\best.pt")

# Set your confidence threshold (0.0 to 1.0)
CONF_THRESHOLD = 0.5  # only show detections above 50% confidence

# Open the webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() # A frame was successfully read ret gets true
    if not ret:
        break

    # Run YOLO inference on the frame with confidence threshold
    results = model(frame, conf=CONF_THRESHOLD)
    
    # Print detected classes and confidence
    for box in results[0].boxes:
        class_id = int(box.cls[0])          # Class index
        class_name = model.names[class_id]  # Class name
        confidence = float(box.conf[0])     # Confidence score

        print(f"Class: {class_name}, Confidence: {confidence:.2f}")
        
    # Draw bounding boxes on the frame
    annotated_frame = results[0].plot()

    # Show the result
    cv2.imshow("YOLOv11 Webcam", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
