from ultralytics import YOLO

# Load detect model
# model = YOLO(r".\ultralytics\cfg\models\v8\yolov8.yaml")  # build a new model from YAML
# model = YOLO(r".\ultralytics\cfg\models\v8\yolov8_CA.yaml")  # build a new model from YAML
# model = YOLO(r'.\ultralytics\cfg\models\v8\yolov8_CA.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# load segment model
model = YOLO(r".\ultralytics\cfg\models\v8\yolov8.yaml").load('yolov8n.pt')  # build a new model from YAML
# model = YOLO(r".\ultralytics\cfg\models\v8\yolov8_CA.yaml")  # build a new model from YAML
# Train the model
model.train(data=r".\datasets\face.yaml", epochs=100, batch=50, imgsz=640, workers=0, device='0')
