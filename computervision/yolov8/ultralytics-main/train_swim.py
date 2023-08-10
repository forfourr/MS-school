from ultralytics import YOLO
if __name__ == "__main__":
    model = YOLO('yolov8s.pt')
    model.train(data='C:/Users/labadmin/MS/MS-school/computervision/yolov8/ultralytics-main/ultralytics/cfg/datasets/swim_yaml.yaml', epochs=20,batch=34, degrees=5, lrf=0.025)


