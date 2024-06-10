import cv2
from ultralytics import YOLO
img_pth = "train/images/aeolis_54_5h_png.rf.762ca3bfc3df592bf8370a8eb03bbe3b.jpg"
model = YOLO("runs/detect/train2/weights/best.pt")
results = model(source=img_pth)
res_plotted = results[0].plot()
cv2.imshow("result", res_plotted)
cv2.waitKey(0)

