import cv2
import os
import numpy as np
from datetime import datetime
from pathlib import Path

root = Path(__file__).resolve().parent
img = cv2.imread(str(root / '00000341.jpg'))  # 读取原图
img_disp = img.copy()
font_face,font_scale,thickness=cv2.FONT_HERSHEY_SIMPLEX,0.5,1
back_ground = 1  

points = []          # 保存所有标注点 (tag, x, y, B, G, R)

def mouseHandler(event,x,y,flags,param):
    color = (0, 0, 200) if back_ground == 1 else (200, 0, 0)
    tag = 'B' if back_ground == 1 else 'P'
    if event == cv2.EVENT_LBUTTONDBLCLK:
        b, g, r = img[y, x]                 # 从原图取像素
        points.append((tag, x, y, int(b), int(g), int(r)))   
        print(tag, x, y, img[y,x])
        cv2.circle(img_disp, (x, y), 4, color, -1)
        cv2.imshow('win',img_disp)

cv2.namedWindow('win')
cv2.setMouseCallback('win',mouseHandler)
cv2.imshow('win',img_disp)

while True:
    key = cv2.waitKey(10) & 0xFF
    if key == 27:          # Esc 退出
        break
    elif key == ord('1'):
        back_ground = 1
    elif key == ord('2'):
        back_ground = 2

# 退出时保存
if points:
    data_dir = root / 'data'
    data_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = data_dir / f'dot_{timestamp}.npz'
    filename_img = os.path.join('data', f'dot_{timestamp}.png')

    cv2.imwrite(filename_img, img_disp)

    a = np.array(points, dtype=object)
    np.savez(filename,
             tag=a[:, 0].astype(str),
             xy=a[:, 1:3].astype(int),
             bgr=a[:, 3:6].astype(int))
    print(f'已保存 {len(points)} 个点到 {filename}')
else:
    print('没有标注点，跳过保存')