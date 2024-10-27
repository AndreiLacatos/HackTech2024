import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pandas as pd
from datetime import datetime
import numpy as np
import os

model = YOLO('C:\\Users\\David\\Desktop\\models\\2030_26.10\\best.pt')

# Initialize Deep SORT tracker with adjusted parameters
tracker = DeepSort(max_age=120, n_init=20, max_cosine_distance=0.35, nn_budget=300)

# Excel export function
def save_to_excel(data, filename):
    df = pd.DataFrame(data, columns=['ID', 'Type', 'Start Date', 'End Date'])
    df.to_excel(filename, index=False)

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

def process_video(video_path, output_excel_name):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return False

    tracked_objects = {}
    excel_data = []

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
            
            # Record start date and initialize tracking information if new object
            if track_id not in tracked_objects:
                tracked_objects[track_id] = {
                    "type": label,
                    "start_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "end_date": None,
                    "last_bbox": current_bbox,
                    "bbox_history": [],
                    "velocity": (0, 0),
                    "frame_count": 1
                }
            else:
                # Update end_date to the current time for active tracks
                tracked_objects[track_id]["end_date"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                tracked_objects[track_id]["frame_count"] += 1

            # Skip this track if it has been visible for less than 10 frames
            if tracked_objects[track_id]["frame_count"] < 10:
                continue

            # Calculate velocity based on previous position
            if len(tracked_objects[track_id]["bbox_history"]) >= 1:
                prev_bbox = tracked_objects[track_id]["bbox_history"][-1]
                delta_t = 0.033  # Assuming 1 frame per time unit, adjust if using real time
                velocity = calculate_velocity(prev_bbox, current_bbox, delta_t)
                tracked_objects[track_id]["velocity"] = velocity

            # Smooth the bounding box position
            smoothed_bbox = smooth_bbox(tracked_objects[track_id]["bbox_history"], current_bbox)
            tracked_objects[track_id]["last_bbox"] = smoothed_bbox

            x1, y1, x2, y2 = smoothed_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"ID {track_id} - {label}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            current_frame_ids.add(track_id)

        # Display object count
        count_text = f"Containers Detected: {len([obj for obj in tracked_objects.values() if obj['frame_count'] >= 10])}"
        cv2.putText(frame, count_text, (1000, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imshow("Containers Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return False

    # Populate excel_data with start and end times for each tracked object
    for track_id, obj_data in tracked_objects.items():
        if obj_data["frame_count"] >= 10:  # Only save data for objects that were detected for 10 or more frames
            excel_data.append([
                track_id,
                obj_data["type"],
                obj_data["start_date"],
                obj_data["end_date"]
            ])

    # Save to Excel
    save_to_excel(excel_data, output_excel_name)
    
    cap.release()
    cv2.destroyAllWindows()
    return True

# Main loop to prompt for folder path, video name, and process
while True:
    folder_path = input("Enter the path to the folder containing videos: ")
    
    if not os.path.isdir(folder_path):
        print("Invalid folder path. Please try again.")
        continue

    video_name = input("Enter the name of the video file (e.g., video.mp4): ")
    video_path = os.path.join(folder_path, video_name)
    
    if not os.path.isfile(video_path):
        print("Video file not found in the specified folder. Please try again.")
        continue

    excel_folder_path = input("Enter the path where the excel file will be saved: ")
    
    if not os.path.isdir(excel_folder_path):
        print("Invalid folder path. Please try again.")
        continue

    output_excel_name = input("Enter the name for the output Excel file (without file type): ")

    excel_path = os.path.join(excel_folder_path, output_excel_name + ".xlsx")

    # Process the video
    if process_video(video_path, excel_path):
        print("Processing complete. The output has been saved to the specified Excel file.")
    
    # Ask if the user wants to process another video or quit
    choice = input("Press 'n' to load a new video or 'q' to quit: ").lower()
    if choice == 'q':
        print("Exiting the program.")
        break
