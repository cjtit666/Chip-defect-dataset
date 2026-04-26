import os
from PIL import Image


# 定义数据所在的根目录
root_dir = r"C:\Users\86183\Desktop\ultralytics-main\datasets\bvn\labels"

# 定义训练集和验证集的文件夹名称
train_folder = "train"
val_folder = "val"


def check_and_delete_invalid_files(folder_path):
    invalid_files_info = []
    files_to_delete = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                txt_file_path = os.path.join(root, file)
                img_file_path = txt_file_path.replace("labels", "images").replace(".txt", ".jpg")
                try:
                    img = Image.open(img_file_path)
                    img_width, img_height = img.size
                except Exception as e:
                    print(f"无法打开图像文件 {img_file_path}，错误信息：{e}")
                    continue

                with open(txt_file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        data = line.strip().split()
                        if len(data) == 5:
                            category_id = int(data[0])
                            x_center, y_center, width, height = map(float, data[1:])

                            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                                invalid_files_info.append(f"在文件 {txt_file_path} 中发现不符合规范的坐标：{line}，对应的图像文件：{img_file_path}")
                                files_to_delete.append(txt_file_path)
                                files_to_delete.append(img_file_path)

    # 列出所有不符合格式的文件信息
    for info in invalid_files_info:
        print(info)

    # 给出提示信息
    input_str = input("上述列出的是所有不符合YOLO格式的文件信息，确认要删除这些文件吗？按任意键继续删除，按Ctrl+C取消...")

    # 删除文件
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"已成功删除文件：{file_path}")
        except OSError as e:
            print(f"删除文件 {file_path} 时出错：{e}")


# 遍历训练集文件夹
train_folder_path = os.path.join(root_dir, train_folder)
check_and_delete_invalid_files(train_folder_path)

# 遍历验证集文件夹
val_folder_path = os.path.join(root_dir, val_folder)
check_and_delete_invalid_files(val_folder_path)