import matplotlib.pyplot as plt
import matplotlib.patches as patches
from test import calculate_iou

def plot_boxes(box1, box2):
    fig, ax = plt.subplots()

    # box1
    box1_rect = patches.Rectangle((box1[0], box1[1]), box1[2]- box1[1], box1[3]-box1[1],
                                  linewidth = 1, edgecolor ='r', facecolor='none')

    ax.add_patch(box1_rect)

    # box2
    box2_rect = patches.Rectangle((box2[0], box2[1]), box2[2]- box2[1], box2[3]-box2[1],
                                  linewidth = 1, edgecolor ='g', facecolor='none')
    ax.add_patch(box2_rect)


    # 겹치는 영역
    xmin_inter = max(box1[0], box2[0])
    ymin_inter = max(box1[1], box2[1])
    xmax_inter = min(box1[2], box2[2])
    ymax_inter = min(box1[3], box2[3])

    if xmax_inter > xmin_inter and ymax_inter > ymin_inter:
        inter_rect = patches.Rectangle((xmin_inter, ymin_inter), xmax_inter-xmin_inter, ymax_inter- ymin_inter,
                                       linewidth = 1, edgecolor ='b', facecolor='none')
        ax.add_patch(inter_rect)

    ax.set_xlim(0,10)
    ax.set_ylim(0,10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


if __name__ == '__main__':
    box1 = [0,0,5,5]
    box2 = [3,3,8,8]

    plot_boxes(box1, box2)

    iou_value = calculate_iou(box1, box2)
    print(iou_value)