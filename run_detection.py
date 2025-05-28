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

# --------- CONFIGURATION ---------
MODEL_PATH = 'yolov8n.pt'  # Path to YOLO model (will auto-download if not present)
IMAGE_PATH = 'data/sample.jpg'  # Default path for image detection
USE_WEBCAM = True  # Flag to toggle between webcam and image detection
CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence score for detections
CAMERA_ID = 0  # Default camera ID (usually 0 for built-in webcam)

# Color configurations
RECTANGLE_COLOR = (255, 255, 255)  # White color for rectangle (BGR format)
TEXT_COLOR = (0, 255, 255)  # Yellow color for text (BGR format)

# --------- HELPER FUNCTIONS ---------
def draw_detections(image, results, fps=None):
    """
    Draw bounding boxes and labels on the image using YOLO detection results.
    
    Args:
        image (numpy.ndarray): Input image to draw on
        results (ultralytics.engine.results.Results): YOLO detection results
        fps (float, optional): Frames per second to display
        
    Returns:
        numpy.ndarray: Image with drawn detections
        
    Process:
    1. Draw FPS counter if provided
    2. Process each detection:
       - Filter by confidence threshold
       - Extract bounding box coordinates
       - Get class name and confidence
       - Draw rectangle and label
    """
    # Draw FPS counter in top-left corner
    if fps is not None:
        cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2)
    
    # Process each detected object
    for box in results.boxes:
        # Get confidence score for this detection
        conf = float(box.conf[0])
        
        # Only process detections above threshold
        if conf > CONFIDENCE_THRESHOLD:
            # Extract bounding box coordinates (x1, y1, x2, y2)
            # These represent the top-left and bottom-right corners
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get class ID and convert to class name
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            
            # Create label with class name and confidence score
            label = f"{class_name}: {conf:.2f}"
            
            # Draw white rectangle around detected object
            # Parameters: image, (x1,y1), (x2,y2), color(BGR), thickness
            cv2.rectangle(image, (x1, y1), (x2, y2), RECTANGLE_COLOR, 2)
            
            # Draw yellow label above the rectangle
            # Parameters: image, text, position, font, scale, color, thickness
            cv2.putText(image, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
    return image

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
    cap = cv2.VideoCapture(0)  # Use default camera (MacBook camera)
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
        # The model automatically handles:
        # - Image preprocessing
        # - Model inference
        # - Post-processing
        results = model(frame)
        
        # Draw detection results with FPS
        frame = draw_detections(frame, results[0], fps)
        
        # Display results
        cv2.imshow('YOLO Detection - MacBook Camera', frame)
        
        # Break loop if 'q' is pressed
        # waitKey(1) means wait for 1ms for a key press
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
    
    # Draw detection results on image
    image = draw_detections(image, results[0])
    
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