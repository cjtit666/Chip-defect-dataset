from ultralytics import YOLO

# 加载模型
model = YOLO()

# 训练模型
model.train(data='yolo-bvn.yaml',workers=0,epochs=2,batch=16)
