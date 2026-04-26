import os
import shutil
import random
from collections import defaultdict

# 设置随机种子以确保可重复性
random.seed(42)

# 源目录路径
labels_directory = r'C:\Users\86183\Desktop\ultralytics-main\datasets\bvn\labels'
images_directory = r'C:\Users\86183\Desktop\ultralytics-main\datasets\bvn\images'

# 目标目录路径
sampled_dataset_dir = r'C:\Users\86183\Desktop\ultralytics-main\datasets\bvn_sampled'
sampled_labels_dir = os.path.join(sampled_dataset_dir, 'labels')
sampled_images_dir = os.path.join(sampled_dataset_dir, 'images')

# 创建目标目录结构
for subset in ['train', 'val']:
    os.makedirs(os.path.join(sampled_labels_dir, subset), exist_ok=True)
    os.makedirs(os.path.join(sampled_images_dir, subset), exist_ok=True)

# 排除的类别
excluded_classes = {'4', '6'}

# 用于存储每个子集的有效样本
data = {
    'train': [],
    'val': []
}

# 遍历train和val子文件夹，收集有效样本
for subset in ['train', 'val']:
    labels_folder = os.path.join(labels_directory, subset)
    images_folder = os.path.join(images_directory, subset)

    for filename in os.listdir(labels_folder):
        if filename.endswith('.txt'):
            label_path = os.path.join(labels_folder, filename)
            image_name = filename.replace('.txt', '.jpg')  # 假设图片为 .jpg 格式
            image_path = os.path.join(images_folder, image_name)

            # 检查图片是否存在
            if not os.path.isfile(image_path):
                print(f"警告: 图片文件不存在 {image_path}")
                continue

            # 读取标签文件内容
            with open(label_path, 'r') as file:
                labels = file.readlines()

            # 检查是否包含被排除的类别
            has_excluded_class = False
            for label in labels:
                class_id = label.split()[0]
                if class_id in excluded_classes:
                    has_excluded_class = True
                    break

            if not has_excluded_class:
                data[subset].append({
                    'image_path': image_path,
                    'label_path': label_path,
                    'filename': filename  # 添加文件名以便后续跟踪
                })

# 输出过滤后的样本数量
print("过滤后的样本数量：")
for subset in ['train', 'val']:
    print(f"{subset} 集: {len(data[subset])} 个样本")

# 计算总目标样本数和各子集的目标样本数
total_target = 4000
train_ratio = 0.8
val_ratio = 0.2
train_target = int(total_target * train_ratio)
val_target = total_target - train_target

# 如果原始数据不足以满足目标，则调整目标
train_available = len(data['train'])
val_available = len(data['val'])

if train_available < train_target:
    print(f"警告: 训练集中可用样本数 ({train_available}) 少于目标数 ({train_target})")
    train_target = train_available

if val_available < val_target:
    print(f"警告: 验证集中可用样本数 ({val_available}) 少于目标数 ({val_target})")
    val_target = val_available


# 简化的分层抽样：随机抽取，不严格按照类别分布
def random_sample(samples, target_count):
    return random.sample(samples, min(target_count, len(samples)))


# 对训练集和验证集分别进行随机抽样
sampled_train = random_sample(data['train'], train_target)
sampled_val = random_sample(data['val'], val_target)

print(f"抽样后训练集样本数: {len(sampled_train)}")
print(f"抽样后验证集样本数: {len(sampled_val)}")
print(f"总样本数: {len(sampled_train) + len(sampled_val)}")


# 复制样本到目标目录
def copy_samples(samples, subset):
    for sample in samples:
        # 复制图片
        dest_image_path = os.path.join(sampled_images_dir, subset, os.path.basename(sample['image_path']))
        shutil.copy2(sample['image_path'], dest_image_path)

        # 复制标签
        dest_label_path = os.path.join(sampled_labels_dir, subset, os.path.basename(sample['label_path']))
        shutil.copy2(sample['label_path'], dest_label_path)


# 执行复制
copy_samples(sampled_train, 'train')
copy_samples(sampled_val, 'val')

print("抽样完成，样本已复制到目标目录。")
