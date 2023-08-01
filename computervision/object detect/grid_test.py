# 그리드가 어떻게 객체 탐지 모델의 입력 데이터를 구성하는지

import cv2
import numpy as np


# 이미지 불러오기
image = cv2.imread("computervision/data/pubao.jpg")

grid_size = (30,30)

# create grid
def create_grid(image, grid_size):
    height, width,_ = image.shape
    grid_width, grid_height = grid_size

    grid_image = np.copy(image)
    for x in range(0, width, grid_width):
        cv2.line(grid_image, (x,0), (x,height), (0,255,0),1)
    for y in range(0, height, grid_height):
        cv2.line(grid_image, (0,y), (width,y), (0,255,0),1)
    
    return grid_image

grid_image = create_grid(image,grid_size)

cv2.imshow("original", image)
cv2.imshow("grid image", grid_image)
cv2.waitKey(0)
cv2.destroyAllWindows()