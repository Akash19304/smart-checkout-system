from ultralytics import YOLO
import cv2
import supervision as sv
from utils import get_product_cost


# coordinates for the line
START = sv.Point(360, 0) 
END = sv.Point(360, 720)


model = YOLO('model\yolov8n.pt')


video_path = "test_videos/video4.mp4"
cap = cv2.VideoCapture(video_path)


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


# change the costs of the products according to above classes
product_costs = {39: 10,  
                 76: 15,
                 46: 20,
                 47: 25,
                 49: 20 
                }


# Initialize products with empty dictionaries for each class
products = {class_id: {} for class_id in product_costs}

video_info = sv.VideoInfo.from_video_path(video_path)

with sv.VideoSink(target_path="test_output/output4.mp4", video_info=video_info) as sink:

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model.track(frame, classes=list(product_costs.keys()), persist=True,
                                  save=True, tracker="bytetrack.yaml")

            # get the boxes
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            cls = results[0].boxes.cls.int().cpu().tolist()

            # visualize the results on the frame
            annotated_frame = results[0].plot()
            detections = sv.Detections.from_ultralytics(results[0])

            for box, track_id, class_id in zip(boxes, track_ids, cls):
                x, y, w, h = box

                if START.y < y < END.y and x > START.x:

                    if track_id not in products[class_id]:
                        products[class_id][track_id] = {'tracked': True}

                        cv2.rectangle(annotated_frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)),
                                      (0, 255, 0), 2)

                if START.y < y < END.y and x < START.x:

                    if track_id in products[class_id]:
                        was_tracked = products[class_id][track_id]['tracked']
                        products[class_id][track_id]['tracked'] = False

                        # Subtract the cost if the product was previously tracked
                        if was_tracked:
                            total_cost -= product_costs.get(class_id, 0)

                        cv2.rectangle(annotated_frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)),
                                    (0, 255, 0), 2)

                        print("-"*20)
                        print(f"Class ID: {class_id}")

            cv2.line(annotated_frame, (START.x, START.y), (END.x, END.y), (0, 255, 0), 2)


            total_cost = sum([get_product_cost(products, product_costs, class_id, track_id) 
                              for class_id in product_costs 
                              for track_id in products.get(class_id, {})])
            

            cost_text = f"Total Cost: {total_cost}"
            print(cost_text)

            cv2.putText(annotated_frame, cost_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            sink.write_frame(annotated_frame)

        else:
            break

cap.release()
