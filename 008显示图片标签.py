import os
import cv2
import numpy as np

# 图片地址
image_path = r'C:\Users\86183\Desktop\ultralytics-main\datasets\4000datasets\images\train\b9d745ebdd8b6502cc0093d577f0212b.jpg'

# 数据集根目录
dataset_root = os.path.dirname(os.path.dirname(os.path.dirname(image_path)))

# 构建标签文件所在的目录路径
labels_folder = os.path.join(dataset_root, 'labels', 'train')

# 提取图片文件名（不包含扩展名）
image_filename = os.path.splitext(os.path.basename(image_path))[0]

# 构建对应的标签文件路径
label_path = os.path.join(labels_folder, f'{image_filename}.txt')

# 读取图片
image = cv2.imread(image_path)
height, width, _ = image.shape

# 检查标签文件是否存在
if os.path.exists(label_path):
    # 打开标签文件并逐行读取
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        # 分割每行内容
        parts = line.strip().split()
        # 归一化的中心点坐标和宽高
        x_center = float(parts[1])
        y_center = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        # 将归一化坐标转换为像素坐标
        x1 = int((x_center - w / 2) * width)
        y1 = int((y_center - h / 2) * height)
        x2 = int((x_center + w / 2) * width)
        y2 = int((y_center + h / 2) * height)

        # 绘制边界框（颜色设置为红色，BGR格式下红色为(0, 0, 255)）
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 保存可视化后的图片
output_path = r'C:\Users\86183\Desktop\ultralytics-main\output_image.jpg'
cv2.imwrite(output_path, image)
print(f"可视化后的图片已保存到 {output_path}，请使用系统自带的图像查看器打开。")