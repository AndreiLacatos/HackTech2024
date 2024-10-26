import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pandas as pd
from datetime import datetime

model = YOLO('C:\\Users\\David\\Desktop\\last.pt')

# Initialize Deep SORT tracker with adjusted parameters
tracker = DeepSort(max_age=350,            # Retain ID longer for disappearing objects
                   n_init=10,              # Frames needed to confirm an ID
                   max_cosine_distance=0.2, # Higher similarity threshold to avoid ID reassignment
                   nn_budget=100)

video_path = 'C:\\Users\\David\\Desktop\\240520_062548_062648.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error")
    exit()

tracked_objects = {}
excel_data = []

def save_to_excel(data):
    df = pd.DataFrame(data, columns=['ID', 'Type', 'Start Date', 'End Date'])
    df.to_excel('C:\\Users\\David\\Desktop\\tracked_objects.xlsx', index=False)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = []

    for detection in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = detection[:6].cpu().numpy()
        label = model.names[int(cls)]
        
        #I set a threshold to detect only those objects with a confidence 
        # score higher than 45% to avoid detecting bins at the edges. 
        # We could probably set it a bit higher.
        if conf > 0.45: 
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

    tracks = tracker.update_tracks(detections, frame=frame)
    current_frame_ids = set()

    for track in tracks:
        # Only process confirmed tracks that are actively updated
        # track.is_confirmed() - returns True if the track is confirmed, 
        # meaning the object has been consistently detected over several frames 
        # (determined by the n_init parameter when setting up Deep SORT).
        # If track.is_confirmed() is False, the tracker hasn't yet "trusted" this track as a valid detection, 
        # so it’s likely a false positive or a very brief, unclear detection.

        # track.time_since_update is an attribute that counts the number of frames since 
        # this track was last updated (i.e., since it was last detected).
        # When an object disappears from the frame (or gets occluded), 
        # time_since_update starts incrementing with each new frame where the object is not detected.

        if not track.is_confirmed() or track.time_since_update > 2:
            continue
        
        track_id = track.track_id
        ltrb = track.to_ltrb()  
        label = track.get_det_class()  
        
        # Draw bounding boxes and labels for actively tracked objects
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"ID {track_id} - {label}"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Record the start date when a new object is detected
        if track_id not in tracked_objects:
            start_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            tracked_objects[track_id] = {"type": label, "start_date": start_date, "end_date": None}
            excel_data.append([track_id, label, tracked_objects[track_id]["start_date"], None])
        
        # Update end date for active tracks
        tracked_objects[track_id]["end_date"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Add the ID to the set of currently detected IDs in this frame
        #current_frame_ids.add(track_id)

    # Display the count of detected objects
    count_text = f"Objects Detected: {len(tracked_objects)}"
    cv2.putText(frame, count_text, (1000, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow("Objects", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

save_to_excel(excel_data)

cap.release()
cv2.destroyAllWindows()
