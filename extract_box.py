import cv2
import numpy as np
from datetime import datetime
import os

img = cv2.imread('00000341.jpg') 
img_marks = img.copy()           
img_disp = img_marks.copy()     # 用于显示当前拖拽预览

font_face,font_scale,thickness=cv2.FONT_HERSHEY_SIMPLEX,0.5,1
back_ground = 1  

points = []          # 保存所有标注点 (tag, x, y, B, G, R)


drawing = False
ix, iy = -1, -1

def mouseHandler(event,x,y,flags,param):
    global drawing, ix, iy, img_disp

    color = (0, 0, 200) if back_ground == 1 else (200, 0, 0)
    tag = 'B' if back_ground == 1 else 'P'

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_disp = img_marks.copy()
            cv2.rectangle(img_disp, (ix, iy), (x, y), color, 2)
            cv2.imshow('win', img_disp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # 只在显示图上画框，取样始终读取原图。
        cv2.rectangle(img_marks, (ix, iy), (x, y), color, 2)
        cv2.imshow('win', img_marks)

        # 计算框的区域
        x_min, x_max = min(ix, x), max(ix, x)
        y_min, y_max = min(iy, y), max(iy, y)

        # 提取框内所有像素（加入 step 步长防止点太多导致内存爆炸）
        step = 2  
        for yy in range(y_min, y_max, step):
            for xx in range(x_min, x_max, step):
                b, g, r = img[yy, xx]
                points.append((tag, xx, yy, int(b), int(g), int(r)))
        
        print(f"框选 {tag} 区域: ({x_min},{y_min}) 到 ({x_max},{y_max})，新增点数: {len(range(y_min, y_max, step)) * len(range(x_min, x_max, step))}")


cv2.namedWindow('win')
cv2.setMouseCallback('win', mouseHandler)
cv2.imshow('win', img)

while True:
    key = cv2.waitKey(10) & 0xFF
    if key == 27:          # Esc 退出
        break
    elif key == ord('1'):
        back_ground = 1
        print("切换到: 背景 (B)")
    elif key == ord('2'):
        back_ground = 2
        print("切换到: 前景 (P)")

# 退出时保存
if points:
    os.makedirs('data', exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join('data', f'box_{timestamp}.npz')
    filename_img = os.path.join('data', f'box_{timestamp}.png')

    cv2.imwrite(filename_img, img_marks)

    a = np.array(points, dtype=object)
    np.savez(filename,
             tag=a[:, 0].astype(str),
             xy=a[:, 1:3].astype(int),
             bgr=a[:, 3:6].astype(int))
    print(f'已保存 {len(points)} 个点到 {filename}')
else:
    print('没有标注点，跳过保存')