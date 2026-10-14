import os
import sys
import argparse
import cv2

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def register_from_image(image_path, name, person_id=""):
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
    print(f"✅ Successfully registered {name} ({person_id or 'Auto-ID'}) -> Saved to {dest_path}")

def register_from_webcam(name, person_id=""):
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
    print("👉 Look at the camera and press SPACEBAR to capture photo.")
    print("👉 Press 'q' to cancel.\n")
    
    captured_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        display = frame.copy()
        cv2.putText(display, f"Registering: {name} (Press SPACE to capture)", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
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
        print(f"✅ Successfully captured and registered {name} ({person_id or 'Auto-ID'}) -> Saved to {dest_path}")
    else:
        print("❌ Registration cancelled.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register a person for face and identity recognition.")
    parser.add_argument("--name", type=str, required=True, help="Person Name (e.g. 'John', 'Alice')")
    parser.add_argument("--id", type=str, default="", help="Person ID (e.g. 'EMP-101', 'VIP-007')")
    parser.add_argument("--image", type=str, default="", help="Optional path to existing photo image file")
    parser.add_argument("--webcam", action="store_true", help="Capture face directly from webcam")

    args = parser.parse_args()
    
    if args.image:
        register_from_image(args.image, args.name, args.id)
    elif args.webcam:
        register_from_webcam(args.name, args.id)
    else:
        # Default to webcam capture if no image is passed
        register_from_webcam(args.name, args.id)
