from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np
from collections import defaultdict
import torch

class SmartCheckoutSystem:
    def __init__(self, model_path='model/yolov8s.pt', line_start=(360, 0), line_end=(360, 720)):
        self.START = sv.Point(*line_start)
        self.END = sv.Point(*line_end)
        
        # Product costs dictionary
        self.product_costs = {
            39: 10,  # bottle
            76: 15,  # scissors
            46: 20,  # banana
            47: 25,  # apple
            49: 20,  # orange
            65: 50,  # remote
        }
        
        # Color mapping for different classes
        self.colors = {
            39: (0, 255, 0),    # bottle - green
            76: (255, 0, 0),    # scissors - red
            46: (255, 255, 0),  # banana - yellow
            47: (0, 255, 255),  # apple - cyan
            49: (255, 0, 255),  # orange - magenta
            65: (0, 165, 255),  # remote - orange
        }
        
        # Initialize model with optimized settings
        self.model = self._setup_model(model_path)
        
        # Initialize tracking state with defaultdict
        self.products = defaultdict(lambda: defaultdict(lambda: {'tracked': False}))
        self.total_cost = 0
        
        # Pre-calculate valid classes
        self.valid_classes = list(self.product_costs.keys())
        
        # Initialize detection buffer for smoothing
        self.detection_buffer = []
        self.buffer_size = 3
        
        # Initialize line overlay
        self.line_overlay = None

    def _setup_model(self, model_path):
        model = YOLO(model_path)
        if torch.cuda.is_available():
            model.to('cuda')
        return model

    def _create_line_overlay(self, frame_shape):
        """Create a static overlay for the checkout line"""
        overlay = np.zeros(frame_shape, dtype=np.uint8)
        # Draw a solid line
        cv2.line(overlay, 
                 (self.START.x, self.START.y), 
                 (self.END.x, self.END.y), 
                 (0, 255, 0), 
                 2)
        # Add a semi-transparent background
        cv2.line(overlay, 
                 (self.START.x, self.START.y), 
                 (self.END.x, self.END.y), 
                 (0, 255, 0), 
                 6, 
                 cv2.LINE_AA)
        return overlay

    def setup_video_source(self, source):
        """Setup video source for either webcam or video file"""
        cap = cv2.VideoCapture(source)
        
        # Set buffer size and FPS for webcam only
        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video source: {source}")
            
        return cap, self._get_video_info(cap)

    def _get_video_info(self, cap):
        return sv.VideoInfo(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=int(cap.get(cv2.CAP_PROP_FPS))
        )

    def _smooth_detections(self, new_detections):
        if not isinstance(new_detections, np.ndarray):
            return new_detections
            
        self.detection_buffer.append(new_detections)
        if len(self.detection_buffer) > self.buffer_size:
            self.detection_buffer.pop(0)
            
        if len(self.detection_buffer) >= 2:
            latest = self.detection_buffer[-1]
            prev = self.detection_buffer[-2]
            
            if len(latest) == len(prev):
                latest[:, :4] = 0.6 * latest[:, :4] + 0.4 * prev[:, :4]
                
        return self.detection_buffer[-1]

    def draw_bounding_box(self, frame, det, class_id):
        """Draw a better looking bounding box with product information"""
        x_center, y_center, w, h = det[:4]
        track_id = int(det[4])
        
        # Calculate box coordinates
        x1 = int(x_center - w/2)
        y1 = int(y_center - h/2)
        x2 = int(x_center + w/2)
        y2 = int(y_center + h/2)
        
        color = self.colors.get(class_id, (0, 255, 0))
        
        # Draw filled rectangle with transparency
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Draw solid border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Add product information
        product_names = {
            39: "Bottle",
            76: "Scissors",
            46: "Banana",
            47: "Apple",
            49: "Orange",
            65: "Remote"
        }
        
        label = f"{product_names.get(class_id, 'Unknown')} (${self.product_costs.get(class_id, 0)})"
        
        # Draw background for text
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1-20), (x1 + label_w, y1), color, -1)
        
        # Draw text
        cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw tracking ID
        cv2.putText(frame, f"ID: {track_id}", (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def process_frame(self, frame):
        # Initialize line overlay if not already done
        if self.line_overlay is None:
            self.line_overlay = self._create_line_overlay(frame.shape)

        # Run inference with optimized settings
        with torch.no_grad():
            results = self.model.track(
                frame,
                classes=self.valid_classes,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                conf=0.5,
                iou=0.5
            )

        if not results or not results[0].boxes:
            # Add the static line overlay
            frame = cv2.addWeighted(frame, 1, self.line_overlay, 0.7, 0)
            return frame, self.total_cost

        boxes = results[0].boxes
        if boxes.id is None:
            # Add the static line overlay
            frame = cv2.addWeighted(frame, 1, self.line_overlay, 0.7, 0)
            return frame, self.total_cost

        detections = np.column_stack([
            boxes.xywh.cpu().numpy(),
            boxes.id.cpu().numpy(),
            boxes.cls.cpu().numpy()
        ])

        smoothed_detections = self._smooth_detections(detections)
        
        # Process detections
        centers = smoothed_detections[:, :2]
        mask_within_y = (self.START.y < centers[:, 1]) & (centers[:, 1] < self.END.y)
        
        # Draw bounding boxes and process detections
        for det in smoothed_detections[mask_within_y]:
            x_center = det[0]
            track_id = int(det[4])
            class_id = int(det[5])
            
            if class_id not in self.product_costs:
                continue
                
            # Update tracking and costs
            if x_center > self.START.x:
                if not self.products[class_id][track_id]['tracked']:
                    self.products[class_id][track_id]['tracked'] = True
                    self.total_cost += self.product_costs[class_id]
            else:
                if self.products[class_id][track_id]['tracked']:
                    self.products[class_id][track_id]['tracked'] = False
                    self.total_cost -= self.product_costs[class_id]
            
            # Draw bounding box with product information
            self.draw_bounding_box(frame, det, class_id)

        # Add the static line overlay
        frame = cv2.addWeighted(frame, 1, self.line_overlay, 0.7, 0)
        
        return frame, self.total_cost

    def run(self, source=0):
        """
        Run the smart checkout system on either webcam or video file
        Args:
            source: Integer for webcam index or string for video file path
        """
        cap, video_info = self.setup_video_source(source)
        
        # Reset tracking state for new session
        self.products.clear()
        self.total_cost = 0
        self.detection_buffer.clear()
        self.line_overlay = None
        
        with sv.VideoSink("test_output/output.mp4", video_info) as sink:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_frame, total_cost = self.process_frame(frame)
                
                # Add cost text with background
                cost_text = f"Total Cost: ${total_cost:.2f}"
                (text_w, text_h), _ = cv2.getTextSize(cost_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                cv2.rectangle(annotated_frame, (5, 5), (15 + text_w, 40), (0, 0, 0), -1)
                cv2.putText(
                    annotated_frame,
                    cost_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )
                
                sink.write_frame(annotated_frame)
                cv2.imshow('Smart Checkout System', annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Initialize the checkout system
    checkout_system = SmartCheckoutSystem()
    
    # Example usage:
    # 1. For webcam (default)
    checkout_system.run()
    
    # 2. For video file (uncomment and modify path as needed)
    # checkout_system.run("test_videos/video4.mp4")