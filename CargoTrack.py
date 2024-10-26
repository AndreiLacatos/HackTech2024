import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pandas as pd
from datetime import datetime
import numpy as np

model = YOLO('Path to yolo')

# Initialize Deep SORT tracker with adjusted parameters
tracker = DeepSort(max_age=120, n_init=20, max_cosine_distance=0.35, nn_budget=300)

video_path = 'Path to video'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error")
    exit()

tracked_objects = {}
excel_data = []

def save_to_excel(data):
    df = pd.DataFrame(data, columns=['ID', 'Type', 'Start Date', 'End Date'])
    df.to_excel('Path where the excel file will be saved', index=False)

def smooth_bbox(bbox_list, new_bbox, window_size=10):
    bbox_list.append(new_bbox)
    if len(bbox_list) > window_size:
        bbox_list.pop(0)
    avg_bbox = [int(sum(x) / len(bbox_list)) for x in zip(*bbox_list)]
    return avg_bbox

def calculate_velocity(prev_bbox, current_bbox, delta_t):
    x1, y1, x2, y2 = current_bbox
    prev_x1, prev_y1, prev_x2, prev_y2 = prev_bbox
    velocity_x = (x1 - prev_x1) / delta_t
    velocity_y = (y1 - prev_y1) / delta_t
    return velocity_x, velocity_y

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = []

    for detection in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = detection[:6].cpu().numpy()
        label = model.names[int(cls)]
        
        if conf > 0.5: 
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

    # Apply tracking and NMS
    tracks = tracker.update_tracks(detections, frame=frame)
    current_frame_ids = set()

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 20:
            continue
        
        track_id = track.track_id
        ltrb = track.to_ltrb()
        label = track.get_det_class()
        current_bbox = [ltrb[0], ltrb[1], ltrb[2], ltrb[3]]
        
        # Short-lived track removal: require a minimum of 10 frames
        if track_id in tracked_objects:
            tracked_objects[track_id]["frame_count"] += 1
        else:
            tracked_objects[track_id] = {
                "type": label,
                "start_date": datetime.now(),
                "end_date": None,
                "last_bbox": current_bbox,
                "bbox_history": [],
                "velocity": (0, 0),
                "frame_count": 1
            }

        if tracked_objects[track_id]["frame_count"] < 10:
            continue  # Skip this track if it has been visible for less than 10 frames

        # Calculate velocity based on previous position
        if len(tracked_objects[track_id]["bbox_history"]) >= 1:
            prev_bbox = tracked_objects[track_id]["bbox_history"][-1]
            delta_t = 0.033  # Assuming 1 frame per time unit, adjust if using real time
            velocity = calculate_velocity(prev_bbox, current_bbox, delta_t)
            tracked_objects[track_id]["velocity"] = velocity

        # Smooth the bounding box position
        smoothed_bbox = smooth_bbox(tracked_objects[track_id]["bbox_history"], current_bbox)
        tracked_objects[track_id]["last_bbox"] = smoothed_bbox
        tracked_objects[track_id]["end_date"] = datetime.now()

        x1, y1, x2, y2 = smoothed_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"ID {track_id} - {label}"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        current_frame_ids.add(track_id)

    count_text = f"Objects Detected: {len([obj for obj in tracked_objects.values() if obj['frame_count'] >= 10])}"
    cv2.putText(frame, count_text, (1000, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow("Objects", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Only save data for objects tracked for at least 10 frames
save_to_excel([data for data in excel_data if tracked_objects[data[0]]["frame_count"] >= 10])

cap.release()
cv2.destroyAllWindows()
