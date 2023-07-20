import cv2

"""
SIFT
SIFT 두 객체의 유사성 비교에 사용

"""

PATH = 'C:/Users/labadmin/MS/MS-school/image_processing/image_classificate/data'
#PATH = 'C:/Users/iiile/Vscode_jupyter/MS_school/MS-school/image_processing/image_classificate/data'

## image loader
image01 = cv2.imread(f"{PATH}/food_dataset/train/burger/081.jpg")
image02 = cv2.imread(f"{PATH}/food_dataset/train/burger/082.jpg")
image01 = cv2.resize(image01, (250,250))
image02 = cv2.resize(image02, (250,250))

gray01 = cv2.cvtColor(image01, cv2.COLOR_BGR2GRAY)
gray02 = cv2.cvtColor(image02, cv2.COLOR_BGR2GRAY)

# image rolation
image02_rotated = cv2.rotate(image02, cv2.ROTATE_90_CLOCKWISE)
gray02_rotated = cv2.rotate(gray02, cv2.ROTATE_90_CLOCKWISE)

## SIFT 객체 생성
sift = cv2.SIFT_create()


## 특징점 검출
keypoints1, descriptors1 = sift.detectAndCompute(gray01, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray02, None)


# 키포인트 매칭
matcher = cv2.BFMatcher()
matches = matcher.match(descriptors1, descriptors2)

# 매칭결과 정렬
matches = sorted(matches, key=lambda x: x.distance)

# 상위 10개 결과 출력
for match in matches[:10]:
    print("distance: ", match.distance)
    print("keypoint1: (x=%d, y=%d)" % (int(keypoints1[match.queryIdx].pt[0]), int(keypoints1[match.queryIdx].pt[1])))
    print("keypoint2: (x=%d, y=%d)" % (int(keypoints2[match.trainIdx].pt[0]), int(keypoints2[match.trainIdx].pt[1])))
    print()

matched_img = cv2.drawMatches(image01, keypoints1, image02_rotated, keypoints2, matches[:10], None)
#프레임 출력
cv2.imshow("SIFT", matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()