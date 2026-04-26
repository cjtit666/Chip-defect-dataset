from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 加载预训练的YOLOv8模型
model = YOLO('yolov8n.pt')  # 这里可以根据你实际拥有的模型文件替换，如'yolov8s.pt'等

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

# 打开原始图片
image = Image.open(image_path)
image = np.array(image)

# 创建一个新的图形和坐标轴对象
fig, ax = plt.subplots(1)

# 在坐标轴上显示原始图片
ax.imshow(image)

# 遍历预测结果并绘制检测框和类别标签
for result in results:
    boxes = result.boxes
    for box in boxes:
        # 获取边界框坐标
        xyxy = box.xyxy.cpu().numpy() if box.xyxy.is_cuda else box.xyxy.numpy()
        x1, y1, x2, y2 = xyxy[0].astype(int)

        # 获取类别名称
        class_name = model.names[int(box.cls)]

        # 创建一个矩形补丁对象用于绘制边界框
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')

        # 将矩形补丁添加到坐标轴上
        ax.add_patch(rect)

        # 在边界框上方添加类别名称标签
        ax.text(x1, y1 - 10, class_name, fontsize=12, color='r', ha='left', va='top')

# 隐藏坐标轴刻度
ax.set_xticks([])
ax.set_yticks([])

# 显示图形
plt.show()