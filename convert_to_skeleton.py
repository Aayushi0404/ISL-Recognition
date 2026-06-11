import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import traceback

# ─── PATHS ───────────────────────────────────────────────────────────────────
KAGGLE_DATASET = r"C:\Users\vaayu\Downloads\ISL PROJECT\ISL DATASET\Indian"
SAVE_DIR       = r"C:\Users\vaayu\Downloads\ISL PROJECT\ISL_skeleton_data"

# ─── SETUP ───────────────────────────────────────────────────────────────────
hd  = HandDetector(maxHands=2, detectionCon=0.5)

IMG_SIZE  = 400
converted = 0
failed    = 0

# ─── CREATE SAVE FOLDERS ─────────────────────────────────────────────────────
print("Creating output folders...")
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    os.makedirs(os.path.join(SAVE_DIR, letter), exist_ok=True)

print(f"Source dataset : {KAGGLE_DATASET}")
print(f"Saving to      : {SAVE_DIR}")
print("=" * 60)
print("Strategy: Normalize ALL hand landmarks to fill canvas properly")
print("=" * 60)


# ─── HELPER: Normalize and draw skeleton ─────────────────────────────────────
def normalize_and_draw(white, all_pts, img_size=400, padding=30):
    """
    Normalize ALL hand landmarks together so they fill the canvas.
    This fixes the issue of tiny/misplaced skeletons.

    all_pts: list of landmark lists (1 or 2 hands)
    """
    # ── Collect ALL points from ALL hands ────────────────────────────────────
    all_x = []
    all_y = []
    for pts in all_pts:
        for p in pts:
            all_x.append(p[0])
            all_y.append(p[1])

    if not all_x:
        return False

    # ── Find bounding box of ALL points combined ──────────────────────────────
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    range_x = max_x - min_x
    range_y = max_y - min_y

    # Avoid division by zero
    if range_x == 0: range_x = 1
    if range_y == 0: range_y = 1

    # ── Scale to fill canvas with padding ────────────────────────────────────
    scale_x = (img_size - 2 * padding) / range_x
    scale_y = (img_size - 2 * padding) / range_y
    scale   = min(scale_x, scale_y)   # uniform scale

    # ── Center on canvas ─────────────────────────────────────────────────────
    scaled_w = range_x * scale
    scaled_h = range_y * scale
    offset_x = (img_size - scaled_w) / 2
    offset_y = (img_size - scaled_h) / 2

    # ── Normalize each hand's points ──────────────────────────────────────────
    colors = [(0, 200, 0), (255, 100, 0)]   # GREEN for hand1, BLUE for hand2

    for hand_idx, pts in enumerate(all_pts):
        color = colors[hand_idx % 2]

        # Normalize points
        norm_pts = []
        for p in pts:
            nx = int((p[0] - min_x) * scale + offset_x)
            ny = int((p[1] - min_y) * scale + offset_y)
            norm_pts.append((nx, ny))

        # ── Draw skeleton lines ───────────────────────────────────────────────
        def draw_line(a, b):
            cv2.line(white, norm_pts[a], norm_pts[b], color, 3)

        for t in range(0, 4):   draw_line(t, t + 1)
        for t in range(5, 8):   draw_line(t, t + 1)
        for t in range(9, 12):  draw_line(t, t + 1)
        for t in range(13, 16): draw_line(t, t + 1)
        for t in range(17, 20): draw_line(t, t + 1)
        draw_line(5, 9);   draw_line(9, 13)
        draw_line(13, 17); draw_line(0, 5); draw_line(0, 17)

        # ── Draw landmark dots ────────────────────────────────────────────────
        for pt in norm_pts:
            cv2.circle(white, pt, 3, (0, 0, 255), -1)

    return True


# ─── MAIN LOOP ───────────────────────────────────────────────────────────────
for folder_name in sorted(os.listdir(KAGGLE_DATASET)):
    letter_folder = os.path.join(KAGGLE_DATASET, folder_name)

    if not os.path.isdir(letter_folder):
        continue
    if not folder_name.isalpha() or len(folder_name) != 1:
        print(f"⏭  Skipping: '{folder_name}' (not a letter)")
        continue

    save_letter = folder_name.upper()
    if save_letter not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        continue

    save_folder = os.path.join(SAVE_DIR, save_letter)
    os.makedirs(save_folder, exist_ok=True)

    image_files = [f for f in os.listdir(letter_folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"\nProcessing {save_letter}: {len(image_files)} images")

    count        = 0
    failed_local = 0
    one_hand_saved  = 0
    two_hand_saved  = 0

    for fname in image_files:
        try:
            frame = cv2.imread(os.path.join(letter_folder, fname))
            if frame is None:
                failed += 1
                failed_local += 1
                continue

            # Fresh white canvas
            white = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255

            # ── Detect ALL hands in full frame ────────────────────────────
            hands     = hd.findHands(frame, draw=False, flipType=True)
            hand_list = hands[0] if isinstance(hands, tuple) else hands

            if not hand_list:
                failed += 1
                failed_local += 1
                continue

            # ── Collect landmarks from ALL detected hands ─────────────────
            all_pts = []
            for hand in hand_list:
                pts = hand['lmList']
                # Convert to (x, y) list
                pts_xy = [(p[0], p[1]) for p in pts]
                all_pts.append(pts_xy)

            # ── Normalize ALL hands together and draw ─────────────────────
            # ✅ KEY FIX: all hands normalized together = consistent scale
            success = normalize_and_draw(white, all_pts, IMG_SIZE)

            if not success:
                failed += 1
                failed_local += 1
                continue

            # Track hand counts
            if len(all_pts) >= 2:
                two_hand_saved += 1
            else:
                one_hand_saved += 1

            # Save skeleton image
            cv2.imwrite(os.path.join(save_folder, f"{count}.jpg"), white)
            count     += 1
            converted += 1

            if count % 50 == 0:
                print(f"  {save_letter}: {count} converted...")

        except Exception:
            failed += 1
            failed_local += 1
            print(f"  Error on {fname}:", traceback.format_exc())
            continue

    print(f"  ✅ Done {save_letter}: {count} saved "
          f"| 1-hand: {one_hand_saved} "
          f"| 2-hand: {two_hand_saved} "
          f"| failed: {failed_local}")

# ─── SUMMARY ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CONVERSION COMPLETE!")
print(f"Successfully converted : {converted} images")
print(f"Failed / No hand found : {failed} images")
print(f"Skeleton data saved to : {SAVE_DIR}")
print("=" * 60)
print("Next step — delete isl_model.h5 and run train_isl_model.py!")
