from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.track(
    source="sample_data/videoplayback.mp4",
    tracker="bytetrack.yaml",
    persist=True,
    save=True,
    show=False,
    classes=[0],
    conf=0.5
)

print("Tracking Complete")