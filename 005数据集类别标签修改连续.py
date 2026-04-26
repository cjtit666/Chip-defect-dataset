import os


def modify_labels(sampled_dataset_dir):
    # 标签目录路径
    labels_folder = os.path.join(sampled_dataset_dir, 'labels')

    # 遍历train和val两个子文件夹
    for subset in ['train', 'val']:
        subset_labels_dir = os.path.join(labels_folder, subset)

        if not os.path.isdir(subset_labels_dir):
            print(f"警告: 文件夹不存在 {subset_labels_dir}")
            continue

        # 遍历标签文件夹中的所有 .txt 文件
        for filename in os.listdir(subset_labels_dir):
            if filename.endswith('.txt'):
                label_path = os.path.join(subset_labels_dir, filename)

                # 读取标签文件内容
                with open(label_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()

                modified = False
                new_lines = []

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 0:
                        continue  # 跳过空行
                    class_id = parts[0]

                    if class_id == '7':
                        parts[0] = '4'
                        modified = True
                        print(f"文件 {label_path} 中的类别 7 已更改为 4")

                    new_line = ' '.join(parts) + '\n'
                    new_lines.append(new_line)

                # 如果有修改，写回文件
                if modified:
                    with open(label_path, 'w', encoding='utf-8') as file:
                        file.writelines(new_lines)

    print("所有标签文件已成功修改。")


if __name__ == "__main__":
    # 数据集目录路径
    sampled_dataset_dir = r'C:\Users\86183\Desktop\ultralytics-main\datasets\4000datasets'

    modify_labels(sampled_dataset_dir)
