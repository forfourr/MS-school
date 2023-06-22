import cv2

# 평균 이동 추적을 위한 초기 사각형 설정
track_window = None     #객체 위치 정보
roi_hist = None         #히스토그램 저장 변수
trem_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)        #수렴조건: (이전위치와 현재 위치의 이동벡터 크기 최소 변경값 | 최대 반복회수)


#동영상 파일 열기
cap = cv2.VideoCapture('data/slow_traffic_small.mp4')

#첫 프레임에서 추적할 객체 선택
ret, frame = cap.read()
x, y, w, h = cv2.selectROI('selectROI', frame, False,False)
print("선택한 박스 좌표>>", x, y, w, h)



#추적할 객체 초기 히스토그램 계산
roi = frame[y: y+h, x: x+w]     

#선택한 박스만 따로 보여주는 창 만들기
#cv2.imshow('roi test', roi)
#cv2.waitKey(0)


hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)    #hsv 색상으로 변환해서 저장
roi_hist = cv2.calcHist([hsv_roi], [0],None, [180], [0,180])#추적할 객체의 색상 정보를 그래프로
cv2.normalize(roi_hist, roi_hist,0, 255, cv2.NORM_MINMAX)   #정규화하여 알고리즘 성넝 개선

track_window = (x, y, w, h)



#Mean shift tracking
while True:
    #프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    #추적할 객체 히스토그램 역투영 계산
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)        #histgram계산은 hvs만 가능해서 변환해줌
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)     #역투영 영상

    #평균 이동 알고리즘을 통해 객체 위치 추정
    ret, track_window = cv2.meanShift(dst, track_window, trem_crit)

    #추적 시작 사각형
    x, y, w, h = track_window
    print(track_window)
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2)

    #프레임 출력
    cv2.imshow('mean shift tracking', frame)

    #'q'버튼으로 종료
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
#자원해제
cap.realse()
cv2.destroyAllWindows()