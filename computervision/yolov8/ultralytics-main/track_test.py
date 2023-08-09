from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO


# load yolov8
model = YOLO('yolov8n.pt')

video_path='tracking_test_video.webm'
cap = cv2.VideoCapture(video_path)
print(cap)

track_history = defaultdict(lambda: [])

while cap.isOpened():
    success, frame = cap.read()


    if success:
        results = model.track(frame, persist=True)
        
        boxes = results[0].boxes.xyxy.cpu().tolist()     #tensor값이 정수형으로 나옴
        track_ids = results[0].boxes.id.int().cpu().tolist()

        annotated_frame = results[0].plot()
        for box, track_id in zip(boxes, track_ids):
            x1,y1,x2,y2 = box
            track = track_history[track_id]

            track.append((float(x1), float(y1)))
            if len(track) > 30:
                track.pop(0)

            points = np.hstack(track).astype(np.int32).reshape((-1,1,2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(210,150,180),thickness=5)

        cv2.imshow("tracking.,,", annotated_frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()