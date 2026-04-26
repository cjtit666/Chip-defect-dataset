import numpy as np
import matplotlib.pyplot as plt

# 定义模型名称和各项指标
models = ['YOLOv5', 'YOLOv6', 'YOLOv7', 'YOLOv8', 'YOLOv9', 'YOLO - FD']
metrics = ['             mAP@0.5', 'Parameters', 'FLOPs      ', 'Size']

# 定义各项指标对应的数据
data = [
    [65.7, 9124514, 24.1, 17.6],
    [55.7, 16307010, 44.2, 31.3],
    [46.2, 37223526, 105.2, 71.3],
    [66.1, 11137922, 28.7, 21.4],
    [66.4, 7289730, 27.4, 14.5],
    [70.5, 8341762, 21.6, 16.1]
]

# 对数据进行归一化处理，同时考虑指标的好坏方向
def normalize_data(data):
    normalized_data = []
    for i in range(len(data[0])):
        column = [row[i] for row in data]
        if i == 0:  # mAP，值越大越好
            min_val = min(column)
            max_val = max(column)
            normalized_column = [(x - min_val) / (max_val - min_val) for x in column]
        else:  # Parameters, FLOPs, Size，值越小越好
            min_val = min(column)
            max_val = max(column)
            normalized_column = [(max_val - x) / (max_val - min_val) for x in column]
        normalized_data.append(normalized_column)
    return list(map(list, zip(*normalized_data)))

# 计算角度
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))

# 高级颜色列表
colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51', '#FF0000']  # 将YOLO - FD的颜色改为红色

# 创建雷达图
fig = plt.figure(figsize=(10, 8), dpi=300)  # 将画布的高度稍微增加一点，并提高清晰度
ax = fig.add_subplot(111, polar=True)

# 绘制每个模型的雷达图
normalized_data = normalize_data(data)
for i, model_data in enumerate(normalized_data):
    model_data = np.concatenate((np.array(model_data), [model_data[0]]))
    ax.plot(angles, model_data, color=colors[i], linewidth=2, label=models[i])
    ax.fill(angles, model_data, facecolor='none', edgecolor=colors[i], alpha=0.1)  # 只有边框有颜色

# 设置标签和标题
ax.set_thetagrids(angles[: - 1] * 180 / np.pi, metrics, fontsize=14)  # 增加标签字体大小
ax.set_title('', size=24, pad=20)  # 增加标题字体大小

# 设置网格线
ax.grid(True)

# 添加图例
plt.legend(loc='best', bbox_to_anchor=(1.3, 1.1), title="Models", fontsize=14)  # 增加图例字体大小

# 保存图形
plt.savefig('radar_chart.png', bbox_inches='tight', dpi=300)

# 显示图形
plt.show()
