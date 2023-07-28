import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

## k-means 클러스터를 위한 함수
def kmeans(boxes, k, num_iters=100):
    # (x2-x1)*(y2-y1)
    box_areas = (boxes[:,2]- boxes[:,0])* (boxes[:,3] - boxes[:,1])
    # [1600 3000 3000 7000 3000 6600]
    indices = np.argsort(box_areas)     #크기순 정렬,인덱스 순서를 반환

    clusters= boxes[indices[-k:]]   # 가장 큰 k개 박스 선택
    prev_clusters = np.zeros_like(clusters)
    
    for _ in range(num_iters):
        # 각 박스와 가장 가까운 클러스터 연결
        box_clusers = np.argmin(((boxes[:, None] - clusters[None]) **2).sum(axis=2), axis=1)
    
    
    
    return clusters




if __name__ == '__main__':
    boxes = np.array([[10,10,50,50], [30,40,80,100], [100,90,150,150],
                      [50,60,120,160], [20,30,70,90], [80,70,140,180]])
    
    # 앵커 박스 설정
    k = 5
    anchors = kmeans(boxes, k)