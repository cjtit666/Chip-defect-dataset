import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

# 配置参数
TARGET_SIZE = 640
IMAGE_DIRS = {
    'train': r'C:\Users\86183\Desktop\ultralytics-main\datasets\bvn\images\train',
    'val': r'C:\Users\86183\Desktop\ultralytics-main\datasets\bvn\images\val'
}
LABEL_DIRS = {
    'train': r'C:\Users\86183\Desktop\ultralytics-main\datasets\bvn\labels\train',
    'val': r'C:\Users\86183\Desktop\ultralytics-main\datasets\bvn\labels\val'
}
OUTPUT_IMAGE_DIRS = {
    'train': r'C:\Users\86183\Desktop\ultralytics-main\datasets\bvn\images_resized\train',
    'val': r'C:\Users\86183\Desktop\ultralytics-main\datasets\bvn\images_resized\val'
}
OUTPUT_LABEL_DIRS = {
    'train': r'C:\Users\86183\Desktop\ultralytics-main\datasets\bvn\labels_resized\train',
    'val': r'C:\Users\86183\Desktop\ultralytics-main\datasets\bvn\labels_resized\val'
}

# 创建输出目录
for split in ['train', 'val']:
    os.makedirs(OUTPUT_IMAGE_DIRS[split], exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIRS[split], exist_ok=True)

def resize_and_pad(image, target_size=640):
    """
    缩放并填充图像到目标尺寸。

    参数:
        image (numpy.ndarray): 原始图像。
        target_size (int): 目标图像尺寸（目标为正方形）。

    返回:
        padded_image (numpy.ndarray): 缩放并填充后的图像。
        scale (float): 缩放比例。
        pad_w (int): 水平填充的像素数（左侧和右侧）。
        pad_h (int): 垂直填充的像素数（顶部和底部）。
    """
    h, w = image.shape[:2]
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_size - new_w
    pad_h = target_size - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    padded_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right,
                                      borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded_image, scale, pad_left, pad_top

def process_labels(label_path, output_label_path, scale, pad_left, pad_top, original_w, original_h, target_size=640):
    """
    调整标签坐标以适应缩放和填充后的图像。

    参数:
        label_path (str): 原始标签文件路径。
        output_label_path (str): 处理后标签文件保存路径。
        scale (float): 缩放比例。
        pad_left (int): 水平填充的像素数（左侧）。
        pad_top (int): 垂直填充的像素数（顶部）。
        original_w (int): 原始图像宽度。
        original_h (int): 原始图像高度。
        target_size (int): 目标图像尺寸。
    """
    if not os.path.exists(label_path):
        print(f"标签文件不存在: {label_path}")
        return

    with open(label_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            print(f"标签格式错误，跳过: {line.strip()}")
            continue  # 跳过格式不正确的标签
        class_id, x_center_norm, y_center_norm, width_norm, height_norm = parts
        class_id = int(class_id)
        x_center_norm = float(x_center_norm)
        y_center_norm = float(y_center_norm)
        width_norm = float(width_norm)
        height_norm = float(height_norm)

        # 将归一化坐标转换为绝对坐标
        x_center_abs = x_center_norm * original_w
        y_center_abs = y_center_norm * original_h
        width_abs = width_norm * original_w
        height_abs = height_norm * original_h

        # 应用缩放
        x_center_scaled = x_center_abs * scale
        y_center_scaled = y_center_abs * scale
        width_scaled = width_abs * scale
        height_scaled = height_abs * scale

        # 添加填充偏移
        x_center_padded = x_center_scaled + pad_left
        y_center_padded = y_center_scaled + pad_top

        # 归一化到目标尺寸
        x_center_new = x_center_padded / target_size
        y_center_new = y_center_padded / target_size
        width_new = width_scaled / target_size
        height_new = height_scaled / target_size

        # 确保归一化后的坐标在 [0,1] 之间
        x_center_new = min(max(x_center_new, 0), 1)
        y_center_new = min(max(y_center_new, 0), 1)
        width_new = min(max(width_new, 0), 1)
        height_new = min(max(height_new, 0), 1)

        new_line = f"{class_id} {x_center_new:.6f} {y_center_new:.6f} {width_new:.6f} {height_new:.6f}\n"
        new_lines.append(new_line)

    # 保存新的标签
    with open(output_label_path, 'w') as f:
        f.writelines(new_lines)

# 主处理循环
for split in ['train', 'val']:
    image_paths = glob(os.path.join(IMAGE_DIRS[split], '*.*'))
    for img_path in tqdm(image_paths, desc=f'Processing {split} images'):
        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            print(f"无法读取图像: {img_path}")
            continue

        original_h, original_w = image.shape[:2]

        # 缩放和填充
        padded_image, scale, pad_left, pad_top = resize_and_pad(image, TARGET_SIZE)

        # 保存处理后的图像
        img_filename = os.path.basename(img_path)
        output_img_path = os.path.join(OUTPUT_IMAGE_DIRS[split], img_filename)
        cv2.imwrite(output_img_path, padded_image)

        # 处理对应的标签
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(LABEL_DIRS[split], label_filename)
        output_label_path = os.path.join(OUTPUT_LABEL_DIRS[split], label_filename)

        process_labels(label_path, output_label_path, scale, pad_left, pad_top, original_w, original_h, TARGET_SIZE)

print("所有图像和标签处理完成！")
