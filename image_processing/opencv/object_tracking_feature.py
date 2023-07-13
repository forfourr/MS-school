#특징점 기반 추적 Feature-based tracking
# SIFT, ORB, SURF 등의 알고리즘으로 특징점 검출
#이 특징점들은 다음 프레임에서 찾아지며 객체의 움직임을 추적
'''
단계
1. 첫 프레임을 읽고 픅백 이미지로 변환
2. Shi-Tomasi 코너 검출기 혹은 다른 검출 방법으로 초기 추적 지점 선택
3. 추적할 특징점의 초기 위치 저장
4. 다음 프레임 읽고 흑백 이미지로 변환
5. Lucas-Kandade 광학 흐름 방법으로 이전 프레임과 현재 프레임 사이의 특징점 이동 벡터 계산
6. 이동 벡터를 이용해여 이전 벡터에서 추적한 특징점 위치를 현재 프레임에 맞게 업데이트
    업데이트
    현재 프레임의 흑백이미지와 업데이트 된 추적 시점으로 다음 프레임의 초기 위치 설정
    이전 프레임의 흑백 이미지와 추적한 특징점 좌표 업데이트
    반복
7. 추적 결과를 시각적으로 표시

'''


import cv2

#동영상 파일 열기
cap = cv2.VideoCapture('data/slow_traffic_small.mp4')

#shi-Tomasi 코너 검출기로 파라미터 설정
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
#{'maxCornoers':100, 'qualityLevel':3 ...}

#광학 흐름 프라미터 설정
lk_params = dict(
    winSize = (15,15),      #특징점 주변 윈도우 크기
    maxLevel=2,             #파라미터 레벨수, 클수록 더 넓은 영역에서 추적
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

#첫 프레임 읽기
ret, prev_frame = cap.read()        #프레임 읽기 성공 여부, 이전 프레임
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)    #그레이 스케일 변환

#초기 추적 지점 선택
prev_corners = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
prev_points = prev_corners.squeeze()    
#sqeeze: NumPy 배열에서 크기가 1인 차원을 제거하는 함수

#추적 결과 표시를 위한 색상
color = (0,255,255)




#feature tracking
while True:
    #프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    #현재 프레임 gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Lucas-kandade 광학 흐름 계산
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray,
                                                      prev_points,
                                                      None,
                                                       **lk_params)
    
    #추적 결과 표시
    for i, (prev_point, next_point) in enumerate(zip(prev_points, next_points)):
        x1, y1 = prev_point.astype(int)
        x2, y2 = next_point.astype(int)

        cv2.line(frame, (x1,y1), (x2,y2), color, 2)
        cv2.circle(frame, (x2,y2), 3, color, -1)

    
    #프레임 출력
    cv2.imshow('feature-based tracking', frame)

    #다음 프레임을 위한 변수 업데이트
    prev_gray = gray.copy()
    prev_points = next_points

    #'q' 누르면 종료
    if cv2.waitKey(30)& 0xFF == ord('q'):
        break

#자원 해제
cap.release()
cv2.destroyAllWindows()