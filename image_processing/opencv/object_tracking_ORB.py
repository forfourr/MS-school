#ORB(Oriented Fast and Rotated BRIEF)
#회전에 불변한 특징점 검출 및 설명 제공
#SIFT, SURF보다 빠르지만 덜 정확

import cv2
import numpy as np

#동영상 파일 열기
cap = cv2.VideoCapture('data/slow_traffic_small.mp4')

#ORB객체 생성
orb = cv2.ORB_create()

################################
#특징점 설정
min_keypoint_size = 10       #최소 크기 설정
duplicate_threshold = 10     #중복 특징점 제거 기준 거리


while(True):
    #프레임 읽기
    ret, frame = cap.read()

    if ret == False:
        break

    #그레이 스케일
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #특징점 검출
    keypoints = orb.detect(gray, None)

    ##############################################
    #특징점 검출
    keypoints = orb.detect(gray, None)
    #특징점 크기가 일정 크기 이상인 것만 남기기
    keypoints = [kp for kp in keypoints if kp.size > min_keypoint_size]
    #중복 특점 제거
    mask = np.ones(len(keypoints), dtype=bool)
    for i, kp1 in enumerate(keypoints):
        if mask[i]:
            for j, kp2 in enumerate(keypoints[i+1:]):
                if mask[i+j+1] and np.linalg.norm(np.array(kp1.pt) - np.array(kp2.pt)) < duplicate_threshold:
                    mask[i+j+1] = False

    keypoints = [kp for i, kp in enumerate(keypoints) if mask[i]]


    #특징점 그리기
    frame = cv2.drawKeypoints(frame, keypoints, None, (0, 255, 0), flags=0)

    #프레임 출력
    cv2.imshow('ORB', frame)

    #종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#자원해제
cap.release()
cv2.destroyAllWindows()