from ultralytics import YOLO
model = YOLO("runs/detect/train34/weights/best.pt")
# 打印模型结构摘要，包括参数数量等信息
model.info()

