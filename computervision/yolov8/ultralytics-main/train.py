from ultralytics import YOLO
if __name__ == "__main__":
    model = YOLO('yolov8s.pt')
    model.train(data='C:/Users/labadmin/MS/MS-school/computervision/data/ultralytics-main/ultralytics/cfg/datasets/glass_yaml.yaml', epochs=100,batch=34, degrees=5, lrf=0.025)


