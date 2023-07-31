
import matplotlib.pyplot as plt
import cv2

def draw_boxes_on_images(img_file, anno_file):
    img = cv2.imread(img_file)
    img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # read txt file
    with open(anno_file,'r', encoding='utf-8') as f:
        #한 줄 씩 읽음 -> line 별로
        lines = f.readlines()
        
        for line in lines:
            values = list(map(float, line.strip().split(' ')))
            # map: 요소 출력 str -> float
            class_id = int(values[0])
            x_min, y_min = int(round(values[1])), int(round(values[2]))
            x_max, y_max = int(round(max(values[3], values[5], values[7]))),\
                            int(round(max(values[4], values[6], values[8])))
            # print(class_id, x_min, y_min, x_max, y_max)

            #image bounding box
            cv2.rectangle(img, (x_min,y_min),(x_max,y_max), (255,0,0),2)
            cv2.putText(img, str(class_id), (x_min,y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))

        plt.figure(figsize=(25,25))
        plt.imshow(img)
        plt.show()

if __name__ == '__main__':
    img_path = 'computervision/data/car_load_dataset/train/syn_00000.png'
    anno_file = 'computervision/data/car_load_dataset/train/syn_00000.txt'

    draw_boxes_on_images(img_path, anno_file)