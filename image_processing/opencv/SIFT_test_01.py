import cv2

"""
SIFT
SIFT 키포인트는 이미지에서 특정한 위치와 크기를 가지며, 식별 가능한 특징을 나타냅니다

"""

PATH = 'C:/Users/labadmin/MS/MS-school/image_processing/image_classificate/data'
#PATH = 'C:/Users/iiile/Vscode_jupyter/MS_school/MS-school/image_processing/image_classificate/data'

## image loader
image = cv2.imread(f"{PATH}/food_dataset/train/burger/081.jpg")
image = cv2.resize(image, (250,250))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


## SIFT 객체 생성
sift = cv2.SIFT_create()


## 특징점 검출
keypoints, descriptors = sift.detectAndCompute(gray, None)

#특징점 그리기
frame = cv2.drawKeypoints(image, keypoints, None)

#프레임 출력
cv2.imshow("SIFT", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
