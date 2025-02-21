import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np
import warnings
import cv2

warnings.filterwarnings("ignore")


def draw_boxes(img, boxes):
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        # Draw rectangle
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Put label text
        label = f"{result.names[int(cls)]} {conf:.2f}"
        cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img


if __name__ == "__main__":
    model = YOLO(r"C:\Users\13359\PycharmProjects\Yolo\yolov8\runs\detect\archive_data_all_default_ep100\weights\best.pt")
    img = cv2.imread(r"D:\Test_data\Yolo_data\archive\data_all\images\train\002.jpg")

    result = model(img, conf=0.5, imgsz=640, half=False)[0]
    boxes = result.boxes.data.tolist()

    # Draw boxes on the image
    img_with_boxes = draw_boxes(img.copy(), boxes)
    plt.imshow(img_with_boxes)
    plt.imsave("result.jpg", img_with_boxes)
    plt.show()
