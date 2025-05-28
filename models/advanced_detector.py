import os
import time
import numpy as np
import torch
import torchvision
from typing import Tuple, Optional, Union, List, Dict
import cv2

class AdvancedObjectDetector:
    def __init__(self, 
                 model_name: str = "yolov5s",
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 device: Optional[str] = None):
        """Initialize the advanced object detector.
        
        Args:
            model_name (str): Name of the YOLOv5 model to use
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for NMS
            device (str, optional): Device to run the model on ('cpu' or 'cuda')
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Get model input size
        self.input_size = (640, 640)  # YOLOv5 default input size
        
    def pad_resize_image(self,
                        image: np.ndarray,
                        new_size: Tuple[int, int] = (640, 640),
                        color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, Tuple[float, Tuple[float, float]]]:
        """Resize and pad image while maintaining aspect ratio.
        
        Args:
            image: Input image in BGR format
            new_size: Target size (width, height)
            color: Padding color in BGR format
            
        Returns:
            Tuple containing:
            - Padded and resized image
            - Tuple of (scale_factor, padding)
        """
        h, w = image.shape[:2]
        new_w, new_h = new_size
        
        # Calculate scaling factor
        scale = min(new_w / w, new_h / h)
        
        # Calculate new dimensions
        new_w_scaled, new_h_scaled = int(w * scale), int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w_scaled, new_h_scaled))
        
        # Calculate padding
        dw = new_w - new_w_scaled
        dh = new_h - new_h_scaled
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)
        
        # Add padding
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=color)
        
        # Calculate padding and scale info for later use
        padding = ((top, bottom), (left, right))
        scale_info = (scale, padding)
        
        return padded, scale_info
    
    def non_max_suppression(self,
                          prediction: torch.Tensor,
                          conf_thres: float = 0.25,
                          iou_thres: float = 0.45,
                          classes: Optional[List[int]] = None,
                          agnostic: bool = False,
                          multi_label: bool = False) -> torch.Tensor:
        """Perform Non-Maximum Suppression on detection results.
        
        Args:
            prediction: Raw model predictions
            conf_thres: Confidence threshold
            iou_thres: IoU threshold
            classes: List of classes to keep
            agnostic: Class-agnostic NMS
            multi_label: Allow multiple labels per box
            
        Returns:
            Filtered detections
        """
        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates
        
        # Settings
        max_wh = 4096  # maximum box width and height
        max_det = 300  # maximum number of detections per image
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 10.0  # seconds to quit after
        
        t = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence
            
            # If none remain process next image
            if not x.shape[0]:
                continue
                
            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])
            
            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            
            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
            
            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            
            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            
            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                print(f'WARNING: NMS time limit {time_limit}s exceeded')
                break
        
        return output[0]
    
    def xywh2xyxy(self, x: torch.Tensor) -> torch.Tensor:
        """Convert bounding boxes from [x, y, w, h] to [x1, y1, x2, y2] format.
        
        Args:
            x: Input tensor of shape (n, 4) with boxes in [x, y, w, h] format
            
        Returns:
            Tensor of shape (n, 4) with boxes in [x1, y1, x2, y2] format
        """
        y = torch.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
    
    def scale_coords(self,
                    img1_shape: Tuple[int, int],
                    coords: np.ndarray,
                    img0_shape: Tuple[int, int],
                    scale_info: Optional[Tuple[float, Tuple[Tuple[int, int], Tuple[int, int]]]] = None) -> np.ndarray:
        """Rescale coords from img1_shape to img0_shape.
        
        Args:
            img1_shape: Shape of the resized image
            coords: Coordinates to rescale
            img0_shape: Shape of the original image
            scale_info: Tuple of (scale_factor, padding)
            
        Returns:
            Rescaled coordinates
        """
        if scale_info is None:
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
            pad = ((img1_shape[0] - img0_shape[0] * gain) / 2,
                  (img1_shape[1] - img0_shape[1] * gain) / 2)
        else:
            gain, pad = scale_info
            
        coords[:, [0, 2]] -= pad[1][0]  # x padding
        coords[:, [1, 3]] -= pad[0][0]  # y padding
        coords[:, :4] /= gain
        
        self.clip_coords(coords, img0_shape)
        return coords
    
    def clip_coords(self, boxes: Union[torch.Tensor, np.ndarray], img_shape: Tuple[int, int]) -> None:
        """Clip bounding boxes to image shape.
        
        Args:
            boxes: Bounding boxes to clip
            img_shape: Image shape (height, width)
        """
        if isinstance(boxes, torch.Tensor):
            boxes[:, 0].clamp_(0, img_shape[1])  # x1
            boxes[:, 1].clamp_(0, img_shape[0])  # y1
            boxes[:, 2].clamp_(0, img_shape[1])  # x2
            boxes[:, 3].clamp_(0, img_shape[0])  # y2
        else:  # np.array
            boxes[:, 0].clip(0, img_shape[1], out=boxes[:, 0])  # x1
            boxes[:, 1].clip(0, img_shape[0], out=boxes[:, 1])  # y1
            boxes[:, 2].clip(0, img_shape[1], out=boxes[:, 2])  # x2
            boxes[:, 3].clip(0, img_shape[0], out=boxes[:, 3])  # y2
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Perform object detection on the input image.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            List of detections, each containing:
            - confidence: Detection confidence
            - class_id: Class ID
            - class_name: Class name
            - bbox: Bounding box [x1, y1, x2, y2]
        """
        # Preprocess image
        padded_img, scale_info = self.pad_resize_image(image, self.input_size)
        
        # Convert to RGB and normalize
        img_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float()
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC to CHW
        img_tensor /= 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(img_tensor)
        
        # Apply NMS
        detections = self.non_max_suppression(
            predictions[0],
            conf_thres=self.conf_threshold,
            iou_thres=self.iou_threshold
        )
        
        # Process detections
        results = []
        if len(detections) > 0:
            # Scale coordinates back to original image
            boxes = detections[:, :4].cpu().numpy()
            boxes = self.scale_coords(self.input_size, boxes, image.shape[:2], scale_info)
            
            # Convert to list of dictionaries
            for i, box in enumerate(boxes):
                results.append({
                    'confidence': float(detections[i, 4]),
                    'class_id': int(detections[i, 5]),
                    'class_name': self.model.names[int(detections[i, 5])],
                    'bbox': box.tolist()
                })
        
        return results 