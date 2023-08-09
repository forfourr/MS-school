from ultralytics import YOLO
import xml.etree.ElementTree as ET
import os
import glob
import cv2

model = YOLO('C:/Users/labadmin/MS/MS-school/computervision/data/ultralytics-main/runs/detect/train6/weights/best.pt')
data_path = 'C:/Users/labadmin/MS/MS-school/computervision/data/ultralytics-main/ultralytics/cfg/datasets/glass_dataset/test'
data_path_list = glob.glob(os.path.join(data_path, "*.jpg"))

####  XML 형식의 어노테이션으로 저장

tree = ET.ElementTree()
root = ET.Element('annotations')

id_num = 0
xml_path = './test_glass.xml'

for path in data_path_list:
    names = model.names
    results = model.predict(path, save=False, imgsz=640, conf=0.5)
    boxes = results[0].boxes
    box_info = boxes
    box_xyxy = box_info.xyxy
    cls = box_info.cls

    image = cv2.imread(path)
    img_h, img_w,_ = image.shape
    file_name = os.path.basename(path)

    
    
    xml_frame = ET.SubElement(root, 'image',id ='%d'%id_num, name=file_name,
                              width = '%d'%img_w, height='%d'%img_h)
    """
    ->
    <annotations>
        <image id=0, name='##.jpg', widht=n, height=m>
    </annotations>
    """

    for bbox, class_num in zip (box_xyxy, cls):
        class_num = int(class_num.item())
        class_name_temp = names[class_num]

        '''
        <annotations>
            <image id=0, name='##.jpg', widht=n, height=m>
                <box label='bound1' source = 'manual' occluded='0' xtl='' ytl='' xbr='' ybr='' zorder='0'></box>
        </annotations>
        
        '''
        x1 = int(bbox[0].item())
        y1 = int(bbox[1].item())
        x2 = int(bbox[2].item())
        y2 = int(bbox[3].item())
        ET.SubElement(xml_frame, 'box', label=str(class_name_temp), source='manual',
                      occluded='0', xtl=str(x1), ytl=str(y1), xbr=str(x2), ybr=str(y2), z_order='0')
        
        id_num +=1
        tree._setroot(root)
        tree.write(xml_path, encoding='utf-8')