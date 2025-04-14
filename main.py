from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
import cv2
import os
import json
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

app = FastAPI()

model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

class TrackResult(BaseModel):
    id: int
    bbox: List[int]

class ProcessResponse(BaseModel):
    tracks: List[TrackResult]

@app.get("/process-sample", response_model=ProcessResponse)
def process_sample_video():
    video_path = "videos/sample_video.webm"
    cap = cv2.VideoCapture(video_path)
    results_list = []
    unique_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, classes=[0])
        detections = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        detections_for_sort = [
            (box.tolist(), conf, 'person')
            for box, conf in zip(detections, confidences)
        ]

        tracks = tracker.update_tracks(detections_for_sort, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            results_list.append({"id": track_id, "bbox": [x1, y1, x2, y2]})
            unique_ids.add(track_id)

    cap.release()

    # Write analytics
    analytics_path = "analytics/summary.json"
    with open(analytics_path, "w") as f:
        json.dump({"total_unique_visitors": len(unique_ids)}, f)

    return ProcessResponse(tracks=results_list)

@app.get("/analytics")
def get_analytics():
    with open("analytics/summary.json") as f:
        data = json.load(f)
    return data

@app.get("/")
def root():
    return {"message": "Visitor Tracker with Analytics is running."}