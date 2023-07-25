import torch
import os
import glob
import cv2

model = torch.hub.load('ultralytics/yolov5','custon',
                       path='computervision/Yolo/yolov5-master/runs/exp/weights/best.pt')


print(model)


label_dict = {
  0: "cat",
  1: "chicken",
  2: "cow",
  3: "dog",
  4: "fox",
  5: "goat",
  6: "horse",
  7: "person",
  8: "racoon",
  9: "skunk"
}

img = 'computervision/Yolo/yolov5-master/animals_dataset/images/'
image =cv2.imread(img)
result = model(img, size=640)
bboxes = result.xyxy[0]

for bbox in bboxes:
    x1, y1, x2, y2, conf, cls = bbox
    x1 = int(x1.item())
    x2  = int(x2.item())
    y1 = int(y1.item())
    y2 = int(y2.item())
    conf = conf.item()
    cls = int(cls.item())
    cls_name = label_dict[cls]
    print(x2, y1, x2, y2, conf, cls_name)
    image = cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)


cv2.imshow('test', image)
if cv2.waitKey(0) & 0xFF == ord('q'):
    exit()