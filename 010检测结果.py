import os
import shutil

# 数据集根目录
dataset_root = r'C:\Users\86183\Desktop\ultralytics-main\datasets\4000datasets'
# 目标目录
target_dir = r'C:\Users\86183\Desktop\结果'
os.makedirs(target_dir, exist_ok=True)
os.makedirs(os.path.join(target_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(target_dir, 'labels'), exist_ok=True)

# 类别集合
class_set = set()
# 用于存储每个类别选取的第一张图片路径
selected_images = {}
# 用于存储每个类别选取的第一张标签文件路径
selected_labels = {}

# 遍历训练集标签目录
train_labels_dir = os.path.join(dataset_root, 'labels', 'train')
for label_file in os.listdir(train_labels_dir):
    label_path = os.path.join(train_labels_dir, label_file)
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            class_id = line.strip().split(' ')[0]
            if class_id not in class_set:
                class_set.add(class_id)
                image_name = label_file.replace('.txt', '.jpg')
                image_path = os.path.join(dataset_root, 'images', 'train', image_name)
                if os.path.exists(image_path):
                    selected_images[class_id] = image_path
                    selected_labels[class_id] = label_path
                    break

# 遍历验证集标签目录
val_labels_dir = os.path.join(dataset_root, 'labels', 'val')
for label_file in os.listdir(val_labels_dir):
    label_path = os.path.join(val_labels_dir, label_file)
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            class_id = line.strip().split(' ')[0]
            if class_id not in class_set:
                class_set.add(class_id)
                image_name = label_file.replace('.txt', '.jpg')
                image_path = os.path.join(dataset_root, 'images', 'val', image_name)
                if os.path.exists(image_path):
                    selected_images[class_id] = image_path
                    selected_labels[class_id] = label_path
                    break

# 将选取的图片复制到目标目录的images子目录
for class_id, image_path in selected_images.items():
    image_name = os.path.basename(image_path)
    target_image_path = os.path.join(target_dir, 'images', image_name)
    shutil.copy2(image_path, target_image_path)

# 将选取的标签文件复制到目标目录的labels子目录
for class_id, label_path in selected_labels.items():
    label_name = os.path.basename(label_path)
    target_label_path = os.path.join(target_dir, 'labels', label_name)
    shutil.copy2(label_path, target_label_path)