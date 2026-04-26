import cv2
import time
from ultralytics import YOLO


def calculate_fps(model_path, image_path, num_iterations = 100):
    model = YOLO(model_path)
    image = cv2.imread(image_path)
    total_time = 0
    for _ in range(num_iterations):
        start_time = time.time()
        results = model.predict(image)
        end_time = time.time()
        total_time += end_time - start_time
    avg_time_per_inference = total_time / num_iterations
    fps = 1 / avg_time_per_inference
    return fps


if __name__ == '__main__':
    model_path = r'C:\Users\86183\Desktop\ultralytics-main\runs\detect\train9\weights\best.pt'
    image_path = r'C:\Users\86183\Desktop\ultralytics-main\datasets\4000datasets\images\val\0a4c7c5c54af4d4ed2d3595069c07d7e.jpg'
    fps = calculate_fps(model_path, image_path)
    print(f"FPS: {fps}")
