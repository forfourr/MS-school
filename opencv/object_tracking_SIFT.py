#SIFT 알고리즘




import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('data/slow_traffic_small.mp4')

#Shift 객체 생성
sift = cv2.SIFT_create()

##############################
#특징점 개수 제한 설정
max_keypoints = 100

while True:
    #프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    #그레이 스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #특징점 검출
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    ################################
    #특점 개수 제한
    if len(keypoints) > max_keypoints:
        keypoints = sorted(keypoints, key=lambda x: -x.response)[:max_keypoints]

    #특징점 그리기
    frame = cv2.drawKeypoints(frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #프레임 출력
    cv2.imshow("SIFT", frame)

    #'q'키를 눌러 종료
    if cv2.waitKey(30)& 0xFF == ord('q'):
        break

#자원 해제
cap.release()
cv2.destroyAllWindows()


#SURF(sppeded-up robust reatures)
#SIFT알고리즘을 개선해 계산 속도 빨라 -> 속도 높임