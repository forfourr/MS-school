#칼만필터
#상태 추정에 사용되는 필터링 기술, 객체 추적에도 적용
#초기 예측과 실제 관측을 기반으로 위치와 속도 추정
#예측과 관측간 오차를 고려해 위치를 업데이트
#움직이는 객체 경로 추적, 예측을 통해 불확실성 줄인다

import cv2
import numpy as np

#칼만 필터 초기화
kalman = cv2.KalmanFilter(4,2)      #갈만 필터 객체 생성
#측정행렬 (측정(2)->상태(4)) 2X4
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
#전이 행렬(시간에 따라 상태 백터가 어떻게 변하는지)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                     [0, 1, 0, 1,],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]], np.float32)
#공분산행렬로 잡읍 설정, 불확실성 모델링
kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32) *0.05

#동영상 파일 열기
cap = cv2.VideoCapture('data/slow_traffic_small.mp4')

#첫 프레임에서 추적할 객체 선택
ret, frame = cap.read()
bbox = cv2.selectROI('select roi', frame, False, False)

#객체 추적을 위한 초기 추정 위치 설정/4차원 벡터
kalman.statePre = np.array([[bbox[0]],      #x좌표
                            [bbox[1]],      #y좌표
                            [0],    
                            [0]], np.float32)   #나머지 두 요소를 0으로 초기화<- 초기상태 객체 속도를 알 수 없기때문에

#x, y, w, h = bbox
#print(x==bbox[0])

#칼만 필터 위치 추적 과정
while True:
    #프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break
    
    x, y, w, h = bbox

    #칼만 필터를 사용해 위치 추적
    #corret로 예측값 보정 (객체의 중심 좌표)
    kalman.correct(np.array([[np.float32(x + w /2)],
                            [np.float32(y + h /2)]]))
    kalman.predict()

    #칼만 필터로 추정된 다음 객체 위치
    predicted_bbox = tuple(map(int, kalman.statePost[:2, 0]))

    #추정된 객체 위치를 사각형으로
    cv2.rectangle(frame,
                  (predicted_bbox[0] - w//2, predicted_bbox[1] - h//2),
                  (predicted_bbox[0] + w //2, predicted_bbox[1] + h //2),
                  (0,0,255), 2)
    
    #프레임 출력
    cv2.imshow('Kalman Filter tracking', frame)

    #'q'누르면 종료
    if cv2.waitKey(30)& 0xFF == ord('q'):
        break

#자원 해제
cap.realse()
cv2.destroyAllWindows()