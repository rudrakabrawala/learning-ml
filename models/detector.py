import torch
import torchvision
import numpy as np
from typing import Tuple, List, Dict, Optional
import cv2

class ObjectDetector:
    def __init__(self, model_name: str = "yolov5s", device: Optional[str] = None):
        """Initialize the object detector with a YOLOv5 model.
        
        Args:
            model_name (str): Name of the YOLOv5 model to use (yolov5s, yolov5m, yolov5l, yolov5x)
            device (str, optional): Device to run the model on ('cpu' or 'cuda')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess the input image for model inference.
        
        Args:
            image (np.ndarray): Input image in BGR format (OpenCV format)
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to model's expected size
        image_resized = cv2.resize(image_rgb, (640, 640))
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image_resized).float()
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC to CHW
        image_tensor /= 255.0  # Normalize to [0, 1]
        
        return image_tensor.unsqueeze(0)  # Add batch dimension
        
    def detect(self, image: np.ndarray, conf_threshold: float = 0.25) -> Dict:
        """Perform object detection on the input image.
        
        Args:
            image (np.ndarray): Input image in BGR format
            conf_threshold (float): Confidence threshold for detections
            
        Returns:
            Dict: Dictionary containing detection results
        """
        with torch.no_grad():
            results = self.model(image)
            
        # Process results
        detections = []
        for pred in results.xyxy[0]:  # xyxy format
            if pred[4] >= conf_threshold:  # Check confidence
                detection = {
                    'bbox': pred[:4].cpu().numpy(),  # x1, y1, x2, y2
                    'confidence': float(pred[4]),
                    'class_id': int(pred[5]),
                    'class_name': results.names[int(pred[5])]
                }
                detections.append(detection)
                
        return {
            'detections': detections,
            'original_image': image
        }
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on the image.
        
        Args:
            image (np.ndarray): Original image
            detections (List[Dict]): List of detection dictionaries
            
        Returns:
            np.ndarray: Image with drawn detections
        """
        image_with_boxes = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            class_name = det['class_name']
            
            # Draw bounding box
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(image_with_boxes, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        return image_with_boxes 