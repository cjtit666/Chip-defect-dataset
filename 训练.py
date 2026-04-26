import os
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import gc

# 设置OpenMP环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置训练配置
model_path = 'ultralytics/cfg/models/v9/yolov9s.yaml'  # 更换为你需要的模型配置文件（如yolov8s.yaml、yolov8m.yaml等）
data_path = 'C:\\Users\\86183\\Desktop\\ultralytics-main\\yolo-bvn.yaml'  # 数据配置文件路径
epochs = 200  # 训练轮数
imgsz = 640  # 图片大小

if __name__ == '__main__':
    # 加载YOLOv8模型
    model = YOLO(model_path, verbose=True)

    # 开始训练
    model.train(data=data_path, epochs=epochs, imgsz=imgsz, workers=1)  # 可调整workers数量，建议不超过4,已经由2改为1

    # 训练完成后进行评估
    results = model.val()

    # 输出评估结果
    print(f"Validation Results: {results}")

    # 释放内存
    del model
    gc.collect()
