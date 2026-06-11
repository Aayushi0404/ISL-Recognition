import cv2
from cvzone.HandTrackingModule import HandDetector
import os

DATASET = r"C:\Users\vaayu\Downloads\ISL PROJECT\ISL DATASET\Indian"

hd = HandDetector(maxHands=2, detectionCon=0.5)

print("=" * 65)
print("Analysing dataset — counting 1-hand vs 2-hand images per letter")
print("This will take 5-10 minutes. Please wait...")
print("=" * 65)

results = {}

for letter in sorted(os.listdir(DATASET)):
    folder = os.path.join(DATASET, letter)

    if not os.path.isdir(folder):
        continue
    if not letter.isalpha() or len(letter) != 1:
        continue

    imgs = [f for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    one_hand  = 0
    two_hands = 0
    no_hand   = 0

    # Check every image (or limit to 200 for speed)
    sample = imgs[:200]

    for fname in sample:
        img = cv2.imread(os.path.join(folder, fname))
        if img is None:
            no_hand += 1
            continue

        hands     = hd.findHands(img, draw=False, flipType=True)
        hand_list = hands[0] if isinstance(hands, tuple) else hands

        if not hand_list:
            no_hand += 1
        elif len(hand_list) == 1:
            one_hand += 1
        else:
            two_hands += 1

    results[letter.upper()] = {
        'total'    : len(sample),
        'one_hand' : one_hand,
        'two_hands': two_hands,
        'no_hand'  : no_hand
    }

    # Determine category
    if two_hands > one_hand:
        category = "2-HAND  🤲"
    elif one_hand > two_hands:
        category = "1-HAND  🖐"
    else:
        category = "MIXED   ⚠️"

    print(f"  {letter.upper()} → {category} "
          f"| 1-hand: {one_hand} "
          f"| 2-hand: {two_hands} "
          f"| no hand: {no_hand} "
          f"| total checked: {len(sample)}")

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)

one_hand_letters  = []
two_hand_letters  = []
mixed_letters     = []

for letter, data in results.items():
    if data['two_hands'] > data['one_hand']:
        two_hand_letters.append(letter)
    elif data['one_hand'] > data['two_hands']:
        one_hand_letters.append(letter)
    else:
        mixed_letters.append(letter)

print(f"1-hand letters  : {sorted(one_hand_letters)}")
print(f"2-hand letters  : {sorted(two_hand_letters)}")
print(f"Mixed letters   : {sorted(mixed_letters)}")
print("=" * 65)
print("Share this output with your maam!")
