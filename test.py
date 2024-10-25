import cv2
from ultralytics import YOLO

model = YOLO('/home/gkap/Desktop/Heckathon/Model detectie containere negre/last.pt')


video_path = '/home/gkap/Desktop/Heckathon/Set date/240517/240517_060340_060350.mp4'
cap = cv2.VideoCapture(video_path)


if not cap.isOpened():
    print("Error")
    exit()


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for idx, detection in enumerate(results[0].boxes.data):
        x1, y1, x2, y2, conf, cls = detection[:6].cpu().numpy()
        label = model.names[int(cls)] 

        object_id = f"ID {idx}"
        text = f"{object_id} - {label}"

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Objects", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

