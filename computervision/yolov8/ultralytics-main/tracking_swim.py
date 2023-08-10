import numpy as np
from ultralytics import YOLO
import cv2
from collections import defaultdict

model = YOLO("runs/detect/train3/weights/best.pt")
video_path = 'C:/Users/labadmin/MS/MS-school/computervision/yolov8/ultralytics-main/tracking_test_video.mp4tracking_test_video.mp4'
cap = cv2.VideoCapture(video_path)
track_history = defaultdict(lambda: [])

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True)

        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        annotated_frame = results[0].plot()

        for box, track_id in zip(boxes,track_ids):
            x,y,w,h = box
            track = track_history[track_ids]

            track.append(float(x), float(y))
            if len(track)>30:
                track.pop(0)


            points = np.hstack(track).astype(np.int32).reshape((-1,1,2))
            cv2.polylines(annotated_frame, [points], isClosed=False,
                        color=(230,230,230), thickness=10)
            
        cv2.imshow("yolov8 tracking swimming", annotated_frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
