from ultralytics import YOLO

model = YOLO('yolo26n-pose.pt')
model.export(format='engine', imgsz=864, half=True, device=0, batch=3)