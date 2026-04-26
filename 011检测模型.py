import os
import cv2
from ultralytics import YOLO

# 加载5个模型
model1 = YOLO(r'C:\Users\86183\Desktop\ultralytics-main\runs\detect\train30\weights\best.pt')
model2 = YOLO(r'C:\Users\86183\Desktop\ultralytics-main\runs\detect\train31\weights\best.pt')
model3 = YOLO(r'C:\Users\86183\Desktop\ultralytics-main\runs\detect\train9\weights\best.pt')
model4 = YOLO(r'C:\Users\86183\Desktop\ultralytics-main\runs\detect\train34\weights\best.pt')
model5 = YOLO(r'C:\Users\86183\Desktop\ultralytics-main\runs\detect\train26\weights\best.pt')

# 将模型存储在一个列表中，方便遍历
models = [model1, model2, model3, model4, model5]

# 图片所在文件夹路径
image_folder = r'C:\Users\86183\Desktop\结果\images'
# 保存检测结果的文件夹路径（与图片在同一文件夹）
result_folder = image_folder

# 遍历图片文件夹中的图片
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    if os.path.isfile(image_path) and image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        for index, model in enumerate(models, start=1):
            # 使用当前模型进行检测
            results = model.predict(source=image_path, save=False)
            # 保存当前模型的检测结果图片
            result_image = results[0].plot()
            result_image_name = f"{os.path.splitext(image_name)[0]}_result{index}.jpg"
            result_image_path = os.path.join(result_folder, result_image_name)
            cv2.imwrite(result_image_path, result_image)