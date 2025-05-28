"""
Real-Time Object Detection using YOLOv8
This script implements real-time object detection using the YOLOv8 model.
It supports both image and webcam-based detection with visualization of results.

Key Concepts:
1. YOLO (You Only Look Once): A state-of-the-art object detection algorithm
2. Real-time Processing: Processing video frames as they are captured
3. Computer Vision: Using OpenCV for image processing and visualization
4. Deep Learning: Using pre-trained YOLO model for object detection
"""

# Import required libraries
import cv2  # OpenCV for image processing and visualization
import numpy as np  # NumPy for numerical operations
from ultralytics import YOLO  # YOLOv8 implementation
import sys  # System-specific parameters and functions
import os  # Operating system interface
import time  # For FPS calculation
from collections import deque

# --------- CONFIGURATION ---------
MODEL_PATH = 'yolov8n.pt'  # Path to YOLO model (will auto-download if not present)
IMAGE_PATH = 'data/sample.jpg'  # Default path for image detection
USE_WEBCAM = True  # Flag to toggle between webcam and image detection
CONFIDENCE_THRESHOLD = 0.45  # Minimum confidence score for detections
CAMERA_ID = 0  # Default camera ID (usually 0 for built-in webcam)

# Color configurations
RECTANGLE_COLOR = (255, 255, 255)  # White color for rectangle (BGR format)
TEXT_COLOR = (0, 255, 255)  # Yellow color for text (BGR format)

# Custom object mapping for similar objects
OBJECT_MAPPING = {
    'toothbrush': 'bottle',  # Map toothbrush to bottle when confidence is low
    'wine glass': 'bottle',  # Map wine glass to bottle when confidence is low
}

# Smoothing parameters for bounding boxes
SMOOTHING_WINDOW = 5  # Number of frames to average for smoothing
box_history = {}  # Dictionary to store box history for each object

# --------- HELPER FUNCTIONS ---------
def smooth_boxes(detections, frame_shape):
    """
    Apply temporal smoothing to bounding boxes to reduce jitter.
    
    Args:
        detections: List of detection results
        frame_shape: Shape of the current frame
    
    Returns:
        List of smoothed detection results
    """
    global box_history
    
    current_boxes = {}
    smoothed_detections = []
    
    # Process current detections
    for det in detections:
        for box in det.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            # Apply custom mapping for low confidence detections
            if conf < 0.6:  # Lower threshold for mapping
                cls_name = det.names[cls]
                if cls_name in OBJECT_MAPPING:
                    # Find the mapped class ID
                    for id, name in det.names.items():
                        if name == OBJECT_MAPPING[cls_name]:
                            cls = id
                            break
            
            # Initialize history for new objects
            if cls not in box_history:
                box_history[cls] = deque(maxlen=SMOOTHING_WINDOW)
            
            # Add current box to history
            box_history[cls].append(xyxy)
            
            # Calculate smoothed box coordinates
            if len(box_history[cls]) > 0:
                smoothed_box = np.mean(box_history[cls], axis=0)
                # Ensure box stays within frame boundaries
                smoothed_box[0] = max(0, min(smoothed_box[0], frame_shape[1]))
                smoothed_box[1] = max(0, min(smoothed_box[1], frame_shape[0]))
                smoothed_box[2] = max(0, min(smoothed_box[2], frame_shape[1]))
                smoothed_box[3] = max(0, min(smoothed_box[3], frame_shape[0]))
                
                # Store smoothed box
                current_boxes[cls] = smoothed_box
                smoothed_detections.append((cls, smoothed_box, conf, det.names[cls]))
    
    # Clean up old boxes
    for cls in list(box_history.keys()):
        if cls not in current_boxes:
            box_history.pop(cls, None)
    
    return smoothed_detections

def draw_detections(frame, detections, fps=None):
    """
    Draw bounding boxes and labels on the frame.
    
    Args:
        frame: Input frame
        detections: List of detection results
        fps: Optional FPS value to display
    
    Returns:
        Frame with drawn detections
    """
    # Define colors for different classes (BGR format)
    colors = {
        'person': (0, 255, 0),    # Green
        'bottle': (255, 0, 0),    # Blue
        'cell phone': (0, 0, 255), # Red
        'chair': (255, 255, 0),   # Cyan
        'dining table': (0, 255, 255), # Yellow
        'toothbrush': (255, 0, 0), # Blue (same as bottle)
        'wine glass': (255, 0, 0), # Blue (same as bottle)
    }
    
    # Draw each detection
    for cls, xyxy, conf, cls_name in detections:
        # Get color for this class
        color = colors.get(cls_name, (255, 255, 255))  # Default to white if class not in colors
        
        # Draw box
        cv2.rectangle(frame, 
                     (int(xyxy[0]), int(xyxy[1])), 
                     (int(xyxy[2]), int(xyxy[3])), 
                     color, 2)
        
        # Draw label
        label = f"{cls_name} {conf:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, 
                     (int(xyxy[0]), int(xyxy[1] - label_height - 10)), 
                     (int(xyxy[0] + label_width), int(xyxy[1])), 
                     color, -1)
        cv2.putText(frame, label, 
                    (int(xyxy[0]), int(xyxy[1] - 5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Draw FPS if provided
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

def detect_on_webcam(model):
    """
    Perform real-time object detection using webcam feed.
    
    Args:
        model (YOLO): Loaded YOLO model
        
    Process:
    1. Initialize webcam
    2. Set up FPS calculation
    3. Process each frame:
       - Capture frame
       - Run detection
       - Draw results
       - Display frame
    4. Clean up resources
    """
    # Initialize webcam capture with MacBook camera
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("Error: Could not open MacBook camera.")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width to 1280p
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height to 720p
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS
    
    print("Press 'q' to quit.")
    
    # FPS calculation variables
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break
            
        # Calculate FPS (update every 30 frames)
        frame_count += 1
        if frame_count >= 30:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()
            
        # Perform detection on frame
        results = model(frame, conf=CONFIDENCE_THRESHOLD)
        
        # Apply smoothing to reduce jitter
        smoothed_results = smooth_boxes(results, frame.shape)
        
        # Draw detection results with FPS
        frame = draw_detections(frame, smoothed_results, fps)
        
        # Display results
        cv2.imshow('YOLO Detection - MacBook Camera', frame)
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up resources
    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all OpenCV windows

def detect_on_image(model, image_path):
    """
    Perform object detection on a single image.
    
    Args:
        model (YOLO): Loaded YOLO model
        image_path (str): Path to input image
        
    Process:
    1. Load image
    2. Run detection
    3. Draw results
    4. Display image
    5. Wait for key press
    """
    # Read image from file
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Perform detection
    results = model(image)
    
    # Process detections
    detections = []
    for det in results:
        for box in det.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            detections.append((cls, xyxy, conf, det.names[cls]))
    
    # Draw detection results on image
    image = draw_detections(image, detections)
    
    # Display results
    cv2.imshow('YOLO Detection - Image', image)
    cv2.waitKey(0)  # Wait for any key press
    cv2.destroyAllWindows()  # Clean up windows

# --------- ENTRY POINT ---------
def main():
    """
    Main function to run the object detection pipeline.
    
    Process:
    1. Load YOLO model
    2. Choose detection mode (webcam/image)
    3. Run detection
    """
    print("Loading YOLO model...")
    # Initialize YOLO model (will auto-download if not present)
    model = YOLO(MODEL_PATH)
    print("Model loaded.")
    
    # Run detection based on configuration
    if USE_WEBCAM:
        detect_on_webcam(model)
    else:
        detect_on_image(model, IMAGE_PATH)

# Run main function if script is executed directly
if __name__ == "__main__":
    main() 