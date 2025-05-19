from ultralytics import YOLO

model = YOLO("yolov5n.pt")  # use 'n' (nano) model for speed

model.train(data="data.yaml", epochs=30, imgsz=640)
model = YOLO("runs/detect/train27/weights/best.pt")
results = model(r"C:\Users\Babn Saravanan\Videos\Projects\Naan mudalvan\images\w_r_30_.jpg", show=True)
