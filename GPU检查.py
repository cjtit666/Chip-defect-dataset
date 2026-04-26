import torch
from ultralytics import YOLO
import numpy as np

# 加载预训练的YOLOv8模型
model = YOLO('yolov8n.pt')
if torch.cuda.is_available():
    model = model.to('cuda')

# 要预测的图片路径
image_path = 'C:\\Users\\86183\\Desktop\\ultralytics-main\\ultralytics\\assets\\bus.jpg'

# 进行预测
results = model.predict(source=image_path)

# 遍历预测结果并详细打印相关信息
for result in results:
    print("检测到的目标数量:", len(result.boxes))
    for i, box in enumerate(result.boxes):
        print(f"目标 {i + 1}:")
        print("    类别:", box.cls)
        print("    类别名称:", model.names[int(box.cls)])  # 根据类别索引获取类别名称
        print("    置信度:", box.conf)
        print("    边界框坐标:", box.xyxy)

        # 将边界框坐标转换为整数坐标（方便后续处理或查看）
        xyxy = box.xyxy.cpu().numpy() if box.xyxy.is_cuda else box.xyxy.numpy()
        x1, y1, x2, y2 = xyxy[0].astype(int)
        print("    左上角坐标: (", x1, ",", y1, ")")
        print("    右下角坐标: (", x2, ",", y2, ")")

        print("    边界框宽度:", x2 - x1)
        print("    边界框高度:", y2 - y1)
