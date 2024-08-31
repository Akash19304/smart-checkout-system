from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from ultralytics import YOLO
import cv2
import numpy as np
from fastapi.responses import JSONResponse, FileResponse
import os

app = FastAPI()

model = YOLO('model/yolov8s.pt')

# Coordinates for the line
START = (360, 0)
END = (360, 720)

# YOLO V8 classes dictionary
YOLO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 
    19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 
    29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 
    37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 
    64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 
    73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

product_costs = {
    39: 10  # example product cost for 'bottle'
}

# Initialize products with empty dictionaries for each class
products = {class_id: {} for class_id in product_costs}

@app.post("/process_video/")
async def process_video(file: UploadFile = File(...)):
    try:
        # Save the uploaded video to a temporary file
        video_temp_file = "temp_video.mp4"
        with open(video_temp_file, "wb") as f:
            f.write(await file.read())

        cap = cv2.VideoCapture(video_temp_file)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Invalid video data")

        # Get video information
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Define the codec and create VideoWriter object
        output_video_file = "output_video.mp4"
        out = cv2.VideoWriter(output_video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        total_cost = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, classes=list(product_costs.keys()), persist=True, save=True, tracker="bytetrack.yaml")

            if results[0].boxes is not None and results[0].boxes.xywh is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                track_ids = results[0].boxes.id
                if track_ids is not None:
                    track_ids = track_ids.int().cpu().numpy()
                else:
                    track_ids = np.zeros(len(boxes))  # or handle appropriately if no IDs are assigned
                cls = results[0].boxes.cls.int().cpu().numpy()

                for box, track_id, class_id in zip(boxes, track_ids, cls):
                    x, y, w, h = box
                    x_center = x
                    y_center = y

                    if START[1] < y_center < END[1]:
                        if x_center > START[0]:
                            if track_id not in products[class_id]:
                                products[class_id][track_id] = {'tracked': True}
                        else:
                            if track_id in products[class_id]:
                                was_tracked = products[class_id][track_id]['tracked']
                                products[class_id][track_id]['tracked'] = False

                                if was_tracked:
                                    total_cost -= product_costs.get(class_id, 0)

                total_cost = 0
                for class_id, tracks in products.items():
                    for track_id, track_info in tracks.items():
                        if track_info['tracked']:
                            total_cost += product_costs[class_id]
                            print("total-cost: ", total_cost)

                annotated_frame = results[0].plot()
                cv2.line(annotated_frame, (START[0], START[1]), (END[0], END[1]), (0, 255, 0), 2)

                cost_text = f"Total Cost: {total_cost}"
                cv2.putText(annotated_frame, cost_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                out.write(annotated_frame)

        cap.release()
        out.release()

        # Clean up the temporary video file
        os.remove(video_temp_file)

        cost_breakdown = {YOLO_CLASSES[class_id]: sum(track_info['tracked'] for track_info in tracks.values()) * product_costs[class_id]
                          for class_id, tracks in products.items()}

        download_url = f"http://127.0.0.1:8000/download/{output_video_file}"
        return JSONResponse(content={"total_cost": total_cost, "cost_breakdown": cost_breakdown, "download_url": download_url})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.get("/download/{file_path:path}")
# async def download_file(file_path: str):
#     # Ensure the path is absolute
#     absolute_file_path = os.path.abspath(file_path)
    
#     if os.path.exists(absolute_file_path):
#         return FileResponse(path=absolute_file_path, filename=os.path.basename(absolute_file_path))
#     else:
#         raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
