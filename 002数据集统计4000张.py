import os

# 源目录路径
sampled_dataset_dir = r'C:\Users\86183\Desktop\ultralytics-main\datasets\4000datasets'
sampled_labels_dir = os.path.join(sampled_dataset_dir, 'labels')
sampled_images_dir = os.path.join(sampled_dataset_dir, 'images')

# 用于统计类别个数的字典
train_class_counts = {}
val_class_counts = {}
train_no_defect_count = 0
val_no_defect_count = 0
complete_no_label_count = 0
complete_label_count = 0

# 遍历train和val两个子文件夹
for folder in ['train', 'val']:
    labels_folder = os.path.join(sampled_labels_dir, folder)
    images_folder = os.path.join(sampled_images_dir, folder)

    # 遍历标签文件夹中的所有文件
    for filename in os.listdir(labels_folder):
        if filename.endswith('.txt'):
            label_path = os.path.join(labels_folder, filename)
            image_name = filename.replace('.txt', '.jpg')  # 假设图片为 .jpg 格式
            full_image_path = os.path.join(images_folder, image_name)

            # 检查图片是否存在
            if not os.path.isfile(full_image_path):
                print(f"警告: 图片文件不存在 {full_image_path}")
                continue

            # 读取标签文件内容
            with open(label_path, 'r') as file:
                labels = file.readlines()

            # 统计无缺陷样本
            if not labels or all(line.strip() == '' for line in labels):  # 处理完全为空的情况
                complete_no_label_count += 1
                if folder == 'train':
                    train_no_defect_count += 1
                elif folder == 'val':
                    val_no_defect_count += 1
            else:
                complete_label_count += 1  # 有标签的数量

            # 统计每个类别
            for label in labels:
                class_id = label.split()[0]  # 类ID在每行的开头
                if folder == 'train':
                    if class_id in train_class_counts:
                        train_class_counts[class_id] += 1
                    else:
                        train_class_counts[class_id] = 1
                elif folder == 'val':
                    if class_id in val_class_counts:
                        val_class_counts[class_id] += 1
                    else:
                        val_class_counts[class_id] = 1

# 输出统计结果
print("=== 新数据集统计结果 ===\n")

print("训练集类别数据个数统计：")
if train_class_counts:
    for class_id in sorted(train_class_counts.keys(), key=int):
        count = train_class_counts[class_id]
        print(f"类别 {class_id}: {count}个")
else:
    print("训练集中没有类别标注。")
print(f"训练集无缺陷个数: {train_no_defect_count}个\n")

print("验证集类别数据个数统计：")
if val_class_counts:
    for class_id in sorted(val_class_counts.keys(), key=int):
        count = val_class_counts[class_id]
        print(f"类别 {class_id}: {count}个")
else:
    print("验证集中没有类别标注。")
print(f"验证集无缺陷个数: {val_no_defect_count}个\n")

print(f"完整的芯片表面（有标记框）的个数: {complete_label_count}个")
print(f"完整的芯片表面（没有标记框）的个数: {complete_no_label_count}个")

# 可选：保存统计结果到文本文件
save_path = os.path.join(sampled_dataset_dir, 'statistics.txt')
with open(save_path, 'w') as f:
    f.write("=== 新数据集统计结果 ===\n\n")

    f.write("训练集类别数据个数统计：\n")
    if train_class_counts:
        for class_id in sorted(train_class_counts.keys(), key=int):
            count = train_class_counts[class_id]
            f.write(f"类别 {class_id}: {count}个\n")
    else:
        f.write("训练集中没有类别标注。\n")
    f.write(f"训练集无缺陷个数: {train_no_defect_count}个\n\n")

    f.write("验证集类别数据个数统计：\n")
    if val_class_counts:
        for class_id in sorted(val_class_counts.keys(), key=int):
            count = val_class_counts[class_id]
            f.write(f"类别 {class_id}: {count}个\n")
    else:
        f.write("验证集中没有类别标注。\n")
    f.write(f"验证集无缺陷个数: {val_no_defect_count}个\n\n")

    f.write(f"完整的芯片表面（有标记框）的个数: {complete_label_count}个\n")
    f.write(f"完整的芯片表面（没有标记框）的个数: {complete_no_label_count}个\n")

print(f"\n统计结果已保存到: {save_path}")
