import os

# 指定目录路径
directory = r'C:\Users\86183\Desktop\ultralytics-main\datasets\bvn\labels\val'

# 遍历目录中的所有文件
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        # 读取文件内容
        with open(file_path, 'r') as file:
            content = file.read().strip()

        # 如果内容为 -1，则清空文件
        if content == '-1':
            with open(file_path, 'w') as file:
                file.write('')  # 清空文件内容

print("处理完成！")
