from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Input video
video_path = "sample_data/videoplayback.mp4"

cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "sample_data/output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

person_count = 0

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    results = model(frame)

    current_count = 0

    for result in results:
        for box in result.boxes:

            cls = int(box.cls[0])

            # Person class = 0
            if cls == 0:

                current_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Person",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    person_count = max(person_count, current_count)

    cv2.putText(
        frame,
        f"People: {current_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    out.write(frame)

cap.release()
out.release()

print("Processing Complete")
print("Maximum People Detected:", person_count)
print("Output saved to sample_data/output.mp4")