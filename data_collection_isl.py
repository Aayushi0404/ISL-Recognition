import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import traceback

# ─── PATHS ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
WHITE_PATH = os.path.join(BASE_DIR, "white.jpg")

# ✅ Change this to your ISL_dataset folder path
SAVE_DIR = r"C:\Users\vaayu\Downloads\ISL_dataset"

# ─── CREATE WHITE IMAGE ───────────────────────────────────────────────────────
white_blank = np.ones((400, 400, 3), np.uint8) * 255
cv2.imwrite(WHITE_PATH, white_blank)

# ─── PRE-CREATE ALL LETTER FOLDERS (A–Z only, no digits) ─────────────────────
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    os.makedirs(os.path.join(SAVE_DIR, letter), exist_ok=True)

# ─── SETUP ───────────────────────────────────────────────────────────────────
capture = cv2.VideoCapture(0)
hd  = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

c_dir  = 'A'
offset = 15
step   = 1
flag   = False
suv    = 0

def get_count(letter):
    folder = os.path.join(SAVE_DIR, letter)
    if os.path.exists(folder):
        return len([f for f in os.listdir(folder)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    return 0

count = get_count(c_dir)

print("=" * 50)
print("ISL Data Collection Started")
print("Controls:")
print("  Press 'a' to START/STOP capturing")
print("  Press 'n' to move to NEXT letter")
print("  Press ESC to quit")
print("=" * 50)
print(f"Current Letter: {c_dir} | Images collected: {count}")

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
while True:
    try:
        ret, frame = capture.read()
        if not ret or frame is None:
            continue
        frame = cv2.flip(frame, 1)

        # Fresh white canvas every frame
        white = np.ones((400, 400, 3), np.uint8) * 255

        hands = hd.findHands(frame, draw=False, flipType=True)
        hand_list = hands[0] if isinstance(hands, tuple) else hands

        skeleton1 = None

        if hand_list:
            hand = hand_list[0]
            x, y, w, h = hand['bbox']

            y1 = max(0, y - offset)
            y2 = min(frame.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(frame.shape[1], x + w + offset)
            image = np.array(frame[y1:y2, x1:x2])

            if image.size > 0:
                handz = hd2.findHands(image, draw=False, flipType=True)
                hand_list2 = handz[0] if isinstance(handz, tuple) else handz

                if hand_list2:
                    hand2 = hand_list2[0]
                    pts   = hand2['lmList']

                    os_x = ((400 - w) // 2) - 15
                    os_y = ((400 - h) // 2) - 15

                    def draw_line(a, b):
                        cv2.line(white,
                                 (pts[a][0] + os_x, pts[a][1] + os_y),
                                 (pts[b][0] + os_x, pts[b][1] + os_y),
                                 (0, 255, 0), 3)

                    for t in range(0, 4):  draw_line(t, t + 1)
                    for t in range(5, 8):  draw_line(t, t + 1)
                    for t in range(9, 12): draw_line(t, t + 1)
                    for t in range(13, 16): draw_line(t, t + 1)
                    for t in range(17, 20): draw_line(t, t + 1)
                    draw_line(5, 9);  draw_line(9, 13)
                    draw_line(13, 17); draw_line(0, 5); draw_line(0, 17)

                    for i in range(21):
                        cv2.circle(white,
                                   (pts[i][0] + os_x, pts[i][1] + os_y),
                                   2, (0, 0, 255), 1)

                    skeleton1 = np.array(white)
                    cv2.imshow("Skeleton Preview", skeleton1)

        # Status overlay
        status = "CAPTURING..." if flag else "Press 'a' to start"
        color  = (0, 0, 255) if flag else (255, 0, 0)
        cv2.putText(frame,
                    f"Letter: {c_dir}  Count: {count}  {status}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2, cv2.LINE_AA)
        cv2.imshow("Camera - ISL Collection", frame)

        interrupt = cv2.waitKey(1)

        if interrupt & 0xFF == 27:          # ESC
            print("Exiting...")
            break

        if interrupt & 0xFF == ord('n'):    # Next letter
            c_dir = chr(ord(c_dir) + 1)
            if ord(c_dir) > ord('Z'):
                c_dir = 'A'
            flag  = False
            count = get_count(c_dir)
            print(f"Switched to Letter: {c_dir} | Existing images: {count}")

        if interrupt & 0xFF == ord('a'):    # Toggle capture
            if flag:
                flag = False
                print("Capture STOPPED")
            else:
                suv  = 0
                flag = True
                print(f"Capture STARTED for letter: {c_dir}")

        # ── Save images ───────────────────────────────────────────────────────
        if flag and skeleton1 is not None:
            if suv >= 180:
                flag = False
                print(f"✅ Done collecting 180 images for letter: {c_dir}")
                print("Press 'n' to go to next letter")
            elif step % 3 == 0:
                # ✅ FIX: use count as filename (not count+1)
                save_path = os.path.join(SAVE_DIR, c_dir, f"{count}.jpg")
                cv2.imwrite(save_path, skeleton1)
                count += 1
                suv   += 1
                print(f"  Saved {c_dir}/{count - 1}.jpg  ({suv}/180)")

            step += 1

    except Exception:
        print("Error:", traceback.format_exc())

capture.release()
cv2.destroyAllWindows()
print("Data collection finished!")
