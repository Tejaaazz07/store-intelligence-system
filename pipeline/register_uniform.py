"""
register_uniform.py
===================
CLI tool for registering the worker uniform profile.

Usage examples:
  # Register by colour name:
  python pipeline/register_uniform.py --color blue
  python pipeline/register_uniform.py --color "red"

  # Register from a text description of the uniform:
  python pipeline/register_uniform.py --describe "dark navy blue polo shirt"
  python pipeline/register_uniform.py --describe "red and white striped t-shirt"

  # Register from a reference image (photo of a staff member or fabric swatch):
  python pipeline/register_uniform.py --image path/to/uniform_photo.jpg

  # Register interactively from webcam snapshot:
  python pipeline/register_uniform.py --webcam

  # Show the currently registered uniform colour:
  python pipeline/register_uniform.py --show

  # List all available colour names:
  python pipeline/register_uniform.py --list

  # Clear all registered uniform profiles:
  python pipeline/register_uniform.py --clear
"""

import sys
import os
import argparse
import cv2

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from uniform_detector import UniformDetector, NAMED_COLORS, _COLOR_SYNONYMS


def capture_from_webcam(save_path="models/uniform_snapshot.jpg"):
    os.makedirs("models", exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return None

    print("\n📸 Webcam opened.")
    print("👉 Make sure a WORKER (in uniform) fills the frame.")
    print("👉 Press SPACEBAR to capture.  Press 'q' to cancel.\n")

    captured = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        disp = frame.copy()
        # Draw torso guide rectangle
        h, w = disp.shape[:2]
        y1, y2 = int(h * 0.25), int(h * 0.65)
        x1, x2 = int(w * 0.15), int(w * 0.85)
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(disp, "Torso region (yellow box) will be used",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(disp, "SPACEBAR = capture   q = cancel",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Register Uniform — Webcam", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:    # spacebar
            captured = frame
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured is not None:
        cv2.imwrite(save_path, captured)
        print(f"✅ Snapshot saved to {save_path}")
        return save_path
    print("❌ Cancelled.")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Register the worker uniform profile for the Store Intelligence System.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --color blue
  %(prog)s --describe "dark navy blue polo shirt"
  %(prog)s --image staff_photo.jpg
  %(prog)s --webcam
  %(prog)s --show
  %(prog)s --list
  %(prog)s --clear
        """
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--color",    type=str,
                     help="Uniform colour name (e.g. blue, red, green, black, white, …)")
    grp.add_argument("--describe", type=str,
                     help="Free-text description of the uniform (e.g. 'dark navy polo shirt')")
    grp.add_argument("--image",    type=str,
                     help="Path to a reference image of a worker or fabric swatch")
    grp.add_argument("--webcam",   action="store_true",
                     help="Capture a snapshot from webcam and use it as reference")
    grp.add_argument("--show",     action="store_true",
                     help="Display the currently registered uniform colour")
    grp.add_argument("--list",     action="store_true",
                     help="List all available colour names and synonyms")
    grp.add_argument("--clear",    action="store_true",
                     help="Clear all registered uniform profiles")

    parser.add_argument("--threshold", type=float, default=0.18,
                        help="Fraction of torso pixels that must match (default 0.18 = 18%%)")

    args = parser.parse_args()
    det = UniformDetector(match_threshold=args.threshold)

    if args.list:
        print("\n🎨 Available colour names:")
        for name in sorted(NAMED_COLORS):
            print(f"   • {name}")
        print("\n🔤 Available synonyms / adjectives:")
        for syn, base in sorted(_COLOR_SYNONYMS.items()):
            print(f"   • {syn:20s} → {base}")
        return

    if args.show:
        print(f"\nℹ️  Current uniform profile: {det.label}")
        print(f"   Number of profiles: {det.profile_count}")
        det.show_color_preview()
        return

    if args.clear:
        det.clear_profiles()
        return

    if args.color:
        ok = det.configure_from_color(args.color)
        if ok:
            print(f"\n🎉 Uniform set to colour: {args.color}")
            det.show_color_preview()

    elif args.describe:
        ok = det.configure_from_description(args.describe)
        if ok:
            print(f"\n🎉 Uniform registered from description: \"{args.describe}\"")
            det.show_color_preview()

    elif args.image:
        ok = det.configure_from_image(args.image)
        if ok:
            print(f"\n🎉 Uniform profile extracted from: {args.image}")
            det.show_color_preview()

    elif args.webcam:
        snap = capture_from_webcam()
        if snap:
            ok = det.configure_from_image(snap)
            if ok:
                print("\n🎉 Uniform profile extracted from webcam snapshot.")
                det.show_color_preview()


if __name__ == "__main__":
    main()
