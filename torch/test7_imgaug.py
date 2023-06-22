import numpy as np
import imgaug.augmenters as iaa
import cv2
import matplotlib.pyplot as plt

def draw(img):
    plt.figure(figsize=(25,25))
    plt.imshow(img)
    plt.show()

image = cv2.imread('data/face2.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_arr = np.array(image)
#image_arr.shape

images= [image_arr, image_arr, image_arr, image_arr]

#1. 아핀변환을 이미지네에 적용
#2D변환의 일종, 이미지 스케일을 조절, 평행이동, 회전
rotate = iaa.Affine(rotate=(-30, 30))
images_aug = rotate(images= images)

#draw(np.hstack(images_aug))

#랜덤하게
rotate_crop = iaa.Sequential([
    iaa.Affine(rotate=(12,12)),
    iaa.Crop(percent=(0, 0.2))
], random_order=True)
image_aug2 = rotate_crop(images= images)

draw(np.hstack(image_aug2))