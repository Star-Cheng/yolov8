import cv2
import numpy as np
from ultralytics import YOLO

import matplotlib.pyplot as plt

color_list = plt.get_cmap("tab20c").colors

if __name__ == "__main__":
    model = YOLO(r"C:\Users\13359\PycharmProjects\Yolo\yolov8\runs\segment\building_ep100_default\weights\best.pt")
    img = cv2.imread(r"D:\Test_data\Yolo_data\Building\data\images\train\0_7.tif")
    result = model(img, conf=0.3, imgsz=640, half=False)[0]
    masks = result.masks
    names = result.names
    boxes = result.boxes.data.tolist()
    h, w = img.shape[:2]

    all_mask = np.zeros((h, w)).astype(np.uint8)
    for i, mask in enumerate(masks.data):
        print(f"mask {i}")
        mask = mask.cpu().numpy().astype(np.uint8)
        mask_resized = cv2.resize(mask, (w, h))

        label = int(boxes[i][5])
        color = np.array(color_list[label][:3]) * 255

        colored_mask = (np.ones((h, w, 3)) * color).astype(np.uint8)
        masked_colored_mask = cv2.bitwise_and(colored_mask, colored_mask, mask=mask_resized)

        mask_indices = mask_resized == 1
        img[mask_indices] = (img[mask_indices] * 0.6 + masked_colored_mask[mask_indices] * 0.4).astype(np.uint8)
        all_mask += mask_resized
    plt.imshow(img)
    plt.savefig("result.jpg",)
    plt.show()
    plt.imshow(all_mask)
    plt.savefig("all_mask.jpg")
    plt.show()
    print("save done")
