#특징점 매칭 알고리즘
#ORB, Brute-force 사용

import cv2

img1=cv2.imread('data/face1.jpg', cv2.IMREAD_GRAYSCALE)
img2=cv2.imread('data/face2.jpg', cv2.IMREAD_GRAYSCALE)

#특징점 검출기 ORB 사용
orb = cv2.ORB_create()

#특징점 검출과 디스크립터 계산
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

#매칭기
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)      #특징점 매칭
matches = sorted(matches, key=lambda x:x.distance)  #매칭 결과 정렬

#매칭결과 그리기
result = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10],
                         None, flags= cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#매칭 결과 출려
cv2.imshow('Match', result)

#퍼센트 계산
num_matches = len(matches)
num_good_matches = sum(1 for m in matches if m.distance < 70)   #거리임계값 설정
matching_percent = (num_good_matches / num_matches)*100

print('Number of matches %.2f%% '% matching_percent)

cv2.waitKey(0)
cv2.destroyAllWindows()