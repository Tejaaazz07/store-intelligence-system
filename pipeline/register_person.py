import os
import sys
import json
import argparse
import cv2

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CONFIG_PATH = os.path.join("models", "workers_config.json")

def save_worker_category(name, person_id, category):
    """Save or update worker category in workers_config.json."""
    os.makedirs("models", exist_ok=True)
    data = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data[name] = {"category": category, "id": person_id or data.get(name, {}).get("id", "Auto-ID")}
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def register_from_image(image_path, name, person_id="", category="Worker"):
    os.makedirs("known_people", exist_ok=True)
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' not found.")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Unable to read image '{image_path}'.")
        return
        
    filename = f"{name}_{person_id}.jpg" if person_id else f"{name}.jpg"
    dest_path = os.path.join("known_people", filename)
    cv2.imwrite(dest_path, img)
    save_worker_category(name, person_id, category)
    print(f"✅ Successfully registered {name} ({person_id or 'Auto-ID'}) as {category} -> Saved to {dest_path}")

def register_from_webcam(name, person_id="", category="Worker"):
    os.makedirs("known_people", exist_ok=True)
    if sys.platform == "win32":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Unable to open webcam.")
        return
        
    print("\n📸 Webcam opened!")
    print(f"👉 Registering: {name} as {category}")
    print("👉 Look at the camera and press SPACEBAR to capture photo.")
    print("👉 Press 'q' to cancel.\n")
    
    captured_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        display = frame.copy()
        cv2.putText(display, f"Registering: {name} [{category}] (Press SPACE to capture)", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imshow("Register Person Face", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 32: # Spacebar
            captured_frame = frame
            break
        elif key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    if captured_frame is not None:
        filename = f"{name}_{person_id}.jpg" if person_id else f"{name}.jpg"
        dest_path = os.path.join("known_people", filename)
        cv2.imwrite(dest_path, captured_frame)
        save_worker_category(name, person_id, category)
        print(f"✅ Successfully captured and registered {name} ({person_id or 'Auto-ID'}) as {category} -> Saved to {dest_path}")
    else:
        print("❌ Registration cancelled.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register a person for face and identity recognition.")
    parser.add_argument("--name", type=str, required=True, help="Person Name (e.g. 'John', 'Alice')")
    parser.add_argument("--id", type=str, default="", help="Person ID (e.g. 'EMP-101', 'EMP-003')")
    parser.add_argument("--category", type=str, default="Worker", help="Worker category (e.g. 'Hacker', 'Hustler', 'Worker')")
    parser.add_argument("--image", type=str, default="", help="Optional path to existing photo image file")
    parser.add_argument("--webcam", action="store_true", help="Capture face directly from webcam")

    args = parser.parse_args()
    
    if args.image:
        register_from_image(args.image, args.name, args.id, args.category)
    elif args.webcam:
        register_from_webcam(args.name, args.id, args.category)
    else:
        register_from_webcam(args.name, args.id, args.category)
