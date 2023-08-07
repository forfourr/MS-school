import os
from xml.etree.ElementTree import parse

class InfraredImageProcess:
    def __init__(self, xml_folder_path):
        self.xml_path = xml_folder_path
        self.label_dict = {'bond':0}

    def find_xml(self):
        all_root = []
        for (path, dir, files) in os.walk(self.xml_path):
            for filename in files:
                ext = os.path.splitext(filename)[-1]    #.xml같은 확장자가 나옴
                if ext == '.xml':
                    root = os.path.join(path, filename)
                    all_root.append(root)
                else:
                    print("could not find .xml file")

        return all_root
    
    def process_img(self):
        xml_dirs = self.find_xml()
        for xml_dir in xml_dirs:
            tree = parse(xml_dir)
            root = tree.getroot()
            ### xml에서 </image>에 대한 정보 다 가져와
            img_metas = root.findall('image')
            for img_meta in img_metas:
                try:
                    # keypoint label list 생성
                    head = []
                    tail = []
                    '''
                    <image id="2" name="IMG_4803_JPG_jpg.rf.f03a06bcdb37c89b3fc2344e61554c4b.jpg" width="1920" height="1080">
                    <points label="head" source="manual" occluded="0" points="921.29,569.82" z_order="0">
                    </points>
                    <box label="bond" source="manual" occluded="0" xtl="770.20" ytl="243.23" xbr="925.94" ybr="533.79" z_order="0">
                    </box>
                    '''

                    # image 이름 가져오기
                    # txt로 저장할 거라 jpg -> txt
                    image_name = img_meta.attrib['name'].replace('.jpg','.txt')
                    img_width = int(img_meta.attrib['width'])
                    img_height = int(img_meta.attrib['height'])
                    label_num = img_meta.attrib['id']
                    print(label_num)

                    # box info
                    box_meta = img_meta.findall('box')
                    point_meta = img_meta.findall('points')

                    
                    ## <points label="tail" source="manual" occluded="0" points="440.67,838.90" z_order="0"></points>
                    for point in point_meta:
                        point_label = point.attrib['label']
                        # points="440.67,838.90"
                        point_x = float(point.attrib['points'].split(',')[0])
                        point_y = float(point.attrib['points'].split(',')[1])
                        
                        if point_label =='head':
                            head = point_x, point_y
                        elif point_label == 'tail':
                            tail = point_x, point_y

                    # <box label="bond" source="manual" occluded="0" xtl="770.20" ytl="243.23" xbr="925.94" ybr="533.79" z_order="0"></box>
                    for box in box_meta:
                        box_label = box.attrib['label']
                        box = [int(float(box.attrib['xtl'])), int(float(box.attrib['ytl'])),
                               int(float(box.attrib['xbt'])), int(float(box.attrib['ybr']))]
                        
                        if box_label == 'ignore':
                            pass
                        
                        box_x = round(((box[0]+box[2])/2)/img_width,6)
                        box_y = round(((box[1]+box[3])/2)/img_height,6)
                        box_w = round((box[2]-box[0])/img_width,6)
                        box_x = round((box[3]-box[1])/img_height,6)

                        head_x_temp = round(head[0]/ img_width, 6)
                        head_y_temp = round(head[0]/ img_height, 6)
                        tail_x_temp = round(tail[0]/ img_width, 6)
                        tail_x_temp = round(tail[0]/ img_height, 6)

                        
                        label_path = 'computervision/data/keypoint_self_dataset/train/labels'
                        os.makedirs(label_path, exist_ok=True)
                        with open(f"{label_path}/{image_name}",'a') as f:
                            f.write(f"{label_num} {box_x:.6f} {box_y:.6f} ")








                except Exception as e:
                    pass



if __name__ == '__main__':
    test = InfraredImageProcess('computervision/data/keypoint_self_dataset/anno/train')
    test.process_img()