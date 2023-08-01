import os
import matplotlib.pyplot as plt
import cv2

def draw_boxes_on_images(img_file, anno_file, img_name):
    img = cv2.imread(img_file)
    img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # read txt file
    with open(anno_file,'r', encoding='utf-8') as f:
        #한 줄 씩 읽음 -> line 별로
        lines = f.readlines()
        
        for line in lines:
            values = line.strip().split(',')
            if values[0] == img_name:
                class_id = float(values[3])
                x_min, y_min = int(round(float(values[4]))), int(round(float(values[5])))
                x_max, y_max = int(round(float(values[6]))), int(round(float(values[7])))
                
            # print(class_id, x_min, y_min, x_max, y_max)

            #image bounding box
                cv2.rectangle(img, (x_min,y_min),(x_max,y_max), (255,0,0),2)
                cv2.putText(img, str(class_id), (x_min,y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))

        plt.figure(figsize=(25,25))
        plt.imshow(img)
        plt.show()

if __name__ == '__main__':
    img_path = 'faster-RCNN/Uno_cards_dataset/train/000244709_jpg.rf.53ce6cbd925fc916030cdbc1910195fb.jpg'
    anno = 'faster-RCNN/Uno_cards_dataset/train/_annotations.csv'
    img_name = os.path.basename(img_path)

    draw_boxes_on_images(img_path, anno, img_name)