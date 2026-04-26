import os
import cv2
import numpy as np
import imghdr


def plot_bounding_boxes(image_path, label_path, save_path):
    # 检查图片路径是否存在
    if not os.path.exists(image_path):
        log_message = f"图片文件不存在: {image_path}"
        print(log_message)
        with open('image_processing_log.txt', 'a') as log_file:
            log_file.write(log_message + '\n')
        return

    # 检查文件大小
    file_size = os.path.getsize(image_path)
    if file_size == 0:
        log_message = f"图片文件大小为0字节，可能不完整: {image_path}"
        print(log_message)
        with open('image_processing_log.txt', 'a') as log_file:
            log_file.write(log_message + '\n')
        return

    # 验证文件是否为有效的图片格式
    img_type = imghdr.what(image_path)
    if img_type not in ['jpeg', 'png', 'jpg']:
        log_message = f"文件格式无效: {image_path}，实际格式为 {img_type}"
        print(log_message)
        with open('image_processing_log.txt', 'a') as log_file:
            log_file.write(log_message + '\n')
        return

    image = cv2.imread(image_path)
    if image is None:
        log_message = f"无法读取图片: {image_path}，请检查文件路径、文件完整性和权限。"
        print(log_message)
        with open('image_processing_log.txt', 'a') as log_file:
            log_file.write(log_message + '\n')
        return

    height, width, _ = image.shape

    if not os.path.exists(label_path):
        log_message = f"标签文件 {label_path} 不存在，跳过此图片。"
        print(log_message)
        with open('image_processing_log.txt', 'a') as log_file:
            log_file.write(log_message + '\n')
        return

    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = list(map(float, line.strip().split()))
            class_id = int(data[0])
            x_center, y_center, box_width, box_height = data[1:]

            # 将归一化的坐标和尺寸转换为实际像素值
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height

            x_min = int(x_center - box_width / 2)
            y_min = int(y_center - box_height / 2)
            x_max = int(x_center + box_width / 2)
            y_max = int(y_center + box_height / 2)

            # 绘制矩形框
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # 在框上方添加类别标签（假设类别ID从0开始）
            cv2.putText(image, str(class_id), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(save_path, image)


image_folder = r'C:\Users\86183\Desktop\结果\images'
label_folder = r'C:\Users\86183\Desktop\结果\labels'
save_folder = r'C:\Users\86183\Desktop\结果\原来'
os.makedirs(save_folder, exist_ok=True)

# 清空日志文件
if os.path.exists('image_processing_log.txt'):
    os.remove('image_processing_log.txt')

for image_name in os.listdir(image_folder):
    if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, image_name)
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(label_folder, label_name)
        save_path = os.path.join(save_folder, image_name)
        plot_bounding_boxes(image_path, label_path, save_path)