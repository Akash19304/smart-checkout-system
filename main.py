from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np
from utils import get_product_cost

# coordinates for the line
START = sv.Point(360, 0)
END = sv.Point(360, 720)

model = YOLO('model/yolov8s.pt')

cap = cv2.VideoCapture(0)  # Use 0 for the primary webcam

"""
- YOLO V8 classes:

{0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 
10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 
19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 
29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 
37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 
46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 
55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 
64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 
73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
"""

product_costs = {
    39: 10,
    76: 15,
    46: 20,
    47: 25,
    49: 20,
    65: 50,
    47: 38
}

# Initialize products with empty dictionaries for each class
products = {class_id: {} for class_id in product_costs}

# Get video information manually
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
video_info = sv.VideoInfo(width=frame_width, height=frame_height, fps=fps)

with sv.VideoSink(target_path="test_output/webcam_output.mp4", video_info=video_info) as sink:
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        results = model.track(frame, classes=list(product_costs.keys()), persist=True, save=True, tracker="bytetrack.yaml")

        # Check if there are any detections
        if results[0].boxes is not None and results[0].boxes.xywh is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id
            if track_ids is not None:
                track_ids = track_ids.int().cpu().numpy()
            else:
                track_ids = np.zeros(len(boxes))  # or handle appropriately if no IDs are assigned
            cls = results[0].boxes.cls.int().cpu().numpy()

            annotated_frame = results[0].plot()
            detections = sv.Detections.from_ultralytics(results[0])

            for box, track_id, class_id in zip(boxes, track_ids, cls):
                x, y, w, h = box
                x_center = x
                y_center = y
                x_left = x_center - w / 2
                y_top = y_center - h / 2
                x_right = x_center + w / 2
                y_bottom = y_center + h / 2

                if START.y < y_center < END.y:
                    if x_center > START.x:
                        if track_id not in products[class_id]:
                            products[class_id][track_id] = {'tracked': True}
                            cv2.rectangle(annotated_frame, (int(x_left), int(y_top)), (int(x_right), int(y_bottom)), (0, 255, 0), 2)
                    else:
                        if track_id in products[class_id]:
                            was_tracked = products[class_id][track_id]['tracked']
                            products[class_id][track_id]['tracked'] = False

                            if was_tracked:
                                total_cost -= product_costs.get(class_id, 0)

                            cv2.rectangle(annotated_frame, (int(x_left), int(y_top)), (int(x_right), int(y_bottom)), (0, 255, 0), 2)
                            print(f"Class ID: {class_id}")

            cv2.line(annotated_frame, (START.x, START.y), (END.x, END.y), (0, 255, 0), 2)

            total_cost = 0
            for class_id, tracks in products.items():
                for track_id, track_info in tracks.items():
                    if track_info['tracked']:
                        total_cost += product_costs[class_id]

            cost_text = f"Total Cost: {total_cost}"
            print(cost_text)

            cv2.putText(annotated_frame, cost_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            sink.write_frame(annotated_frame)

            cv2.imshow('Smart Checkout System', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
