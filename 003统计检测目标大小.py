import os
import glob
from collections import defaultdict
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 定义数据集路径
dataset_path = r"C:\Users\86183\Desktop\ultralytics-main\datasets\4000datasets"
labels_path = os.path.join(dataset_path, "labels")

# 定义训练和验证集的标签文件夹
train_labels = os.path.join(labels_path, "train")
val_labels = os.path.join(labels_path, "val")

# 获取所有标签文件的路径
label_files = glob.glob(os.path.join(train_labels, "*.txt")) + glob.glob(os.path.join(val_labels, "*.txt"))

# 初始化计数器
counts = defaultdict(lambda: {'small': 0, 'medium': 0, 'large': 0})

# 定义面积阈值（基于640x640的输入图像）
SMALL_AREA_THRESHOLD = 0.0025  # 小目标面积阈值
MEDIUM_AREA_THRESHOLD = 0.04    # 中等目标面积阈值

# 遍历所有标签文件
for label_file in label_files:
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # 跳过格式不正确的行
            # YOLO格式：<class> <x_center> <y_center> <width> <height>
            try:
                class_id = parts[0]
                width = float(parts[3])
                height = float(parts[4])
                area = width * height  # 计算归一化面积
                if area < SMALL_AREA_THRESHOLD:
                    counts[class_id]['small'] += 1
                elif area < MEDIUM_AREA_THRESHOLD:
                    counts[class_id]['medium'] += 1
                else:
                    counts[class_id]['large'] += 1
            except ValueError:
                continue  # 跳过无法转换为浮点数的值

# 计算总数
total = sum(count['small'] + count['medium'] + count['large'] for count in counts.values())
small_total = sum(count['small'] for count in counts.values())
medium_total = sum(count['medium'] for count in counts.values())
large_total = sum(count['large'] for count in counts.values())

# 输出统计结果
print(f"总目标数量: {total}")
print(f"小目标数量: {small_total}")
print(f"中等目标数量: {medium_total}")
print(f"大目标数量: {large_total}")

# 按类别输出
for class_id, count in counts.items():
    print(f"类别 {class_id}: 小={count['small']}, 中={count['medium']}, 大={count['large']}")

# 可视化
categories = ['小目标', '中等目标', '大目标']
sizes = [small_total, medium_total, large_total]
colors = ['#66b3ff','#99ff99','#ff9999']

plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=categories, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('目标大小分布')
plt.axis('equal')  # 保证饼图是圆的
plt.show()
