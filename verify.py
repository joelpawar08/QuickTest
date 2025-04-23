from ultralytics import YOLO
model1 = YOLO("280.pt")
print(model1.names)
model2 = YOLO("maggie.pt")
print(model2.names)