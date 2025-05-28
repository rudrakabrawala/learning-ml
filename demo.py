import cv2
import time
from models.advanced_detector import AdvancedObjectDetector

def process_image(image_path: str, detector: AdvancedObjectDetector):
    """Process a single image and display results.
    
    Args:
        image_path: Path to the input image
        detector: Initialized detector instance
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Perform detection
    start_time = time.time()
    detections = detector.detect(image)
    inference_time = time.time() - start_time
    
    # Draw detections
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        conf = det['confidence']
        class_name = det['class_name']
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name}: {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Add inference time
    cv2.putText(image, f"Inference time: {inference_time:.3f}s",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display results
    cv2.imshow("Object Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path: str, detector: AdvancedObjectDetector):
    """Process video stream and display results in real-time.
    
    Args:
        video_path: Path to the input video (0 for webcam)
        detector: Initialized detector instance
    """
    # Open video capture
    cap = cv2.VideoCapture(int(video_path) if video_path.isdigit() else video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_path}")
        return
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform detection
        start_time = time.time()
        detections = detector.detect(frame)
        inference_time = time.time() - start_time
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            class_name = det['class_name']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add FPS
        fps = 1.0 / inference_time
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display results
        cv2.imshow("Object Detection", frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Initialize detector
    detector = AdvancedObjectDetector(
        model_name="yolov5s",
        conf_threshold=0.25,
        iou_threshold=0.45
    )
    
    # Process image
    print("Processing image...")
    process_image("data/sample.jpg", detector)
    
    # Process video/webcam
    print("Processing video/webcam...")
    process_video("0", detector)  # Use "0" for webcam or provide video path

if __name__ == "__main__":
    main() 