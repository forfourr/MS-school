import json
import os
from PIL import Image

# image resize, bounding box scale
def resize_and_scale_bbox(img, bbox, target_size):
    img_width , img_height = img.size

    # resize
    img = img.resize(target_size, Image.LANCZOS)    #고급 리샘플링 알고리즘

    # bbox scale
    x, y, width, height=  bbox
    x_scale = target_size[0]/img_width
    y_scale = target_size[1]/img_height

    x_center = (x+width /2) * x_scale
    y_center = (y+height/2) * y_scale
    scaled_width = width * x_scale
    scaled_height = height * y_scale

    scaled_bbox = (x_center, y_center, scaled_width, scaled_height)
    return img, scaled_bbox


def coco_to_yolo(coco_path, yolo_path,img_path, target_size):
    with open(coco_path, 'r') as f:
        coco_data = json.load(f)

    yolo_data = []
    labels = {category['id']: category['name'] for category in coco_data['categories']}
    
    for image in coco_data['images']:
        image_id = image['id']
        file_name = image['file_name']
        image_path = os.path.join('computervision/data/website_dataset/valid',file_name)
        height = image['height']
        width = image['width']

        img = Image.open(image_path)

        yolo_data =[]
        annotations = [anno for anno in coco_data['annotations'] if anno['image_id'] == image_id]
        # anno
        for anno in annotations:
            category_id = anno['category_id']
            category_name = labels[category_id]
            bbox = anno['bbox']

            img_resized, scaled_bbox = resize_and_scale_bbox(img, bbox, target_size)
            #img_resized, _ = resize_and_scale_bbox(img, [0, 0, width, height], target_size)
            img_resized.save(os.path.join(img_path, f"{os.path.splitext(file_name)[0]}.jpg"))


            x_center, y_center, scaled_width, scaled_height = scaled_bbox
            yolo_label = f"{category_name} {x_center:.6f} {y_center:.6f} {scaled_width:.6f} {scaled_height:.6f}"
            yolo_data.append(yolo_label)

        yolo_json_data = {
            'image_id': image_id,
            'file_name': file_name,
            'width': target_size[0],
            'height': target_size[1],
            'annotations': yolo_data
        }

        # Create a JSON file for each image
        json_file_path = os.path.join(yolo_path, f"{os.path.splitext(file_name)[0]}.txt")
        with open(json_file_path, 'w') as json_file:
            json.dump(yolo_json_data, json_file, indent=4)






if __name__ == '__main__':
    coco_path = 'computervision/data/website_dataset/valid/_annotations.coco.json'
    yolo_path = 'computervision/data/website_dataset/labels/valid'
    img_path = 'computervision/data/website_dataset/images/valid'
    os.makedirs(yolo_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)
    coco_to_yolo(coco_path, yolo_path, img_path, target_size=(416,416))