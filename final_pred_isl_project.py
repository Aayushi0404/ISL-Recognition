# ─── Indian Sign Language — Real-Time Prediction ─────────────────────────────
import numpy as np
import math
import cv2
import os
import traceback
import pyttsx3
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
import enchant
import tkinter as tk
from PIL import Image, ImageTk
from collections import Counter

# ─── PATHS ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
WHITE_PATH = os.path.join(BASE_DIR, "white.jpg")
MODEL_PATH = os.path.join(BASE_DIR, "isl_model.h5")

if not os.path.exists(WHITE_PATH):
    cv2.imwrite(WHITE_PATH, np.ones((400, 400, 3), np.uint8) * 255)

IMG_SIZE = 64
LETTERS  = list(ascii_uppercase)   # A-Z

# ─── CTC CONFIG ──────────────────────────────────────────────────────────────
BLANK_TOKEN          = "_"
CONFIDENCE_THRESHOLD = 0.6
CTC_BUFFER_SIZE      = 40
STABILITY_THRESHOLD  = 0.55  # FIX: lowered from 0.75 to 0.55 — less strict
COOLDOWN_SECONDS     = 1.5

ddd    = enchant.Dict("en_US")
hd     = HandDetector(maxHands=2)
offset = 29


# ─── CTC DECODER ─────────────────────────────────────────────────────────────
def ctc_decode(predictions):
    decoded = []
    prev    = None
    for p in predictions:
        if p == prev:
            continue
        if p != BLANK_TOKEN:
            decoded.append(p)
        prev = p
    return "".join(decoded)


# ─── NORMALIZE AND DRAW SKELETON ─────────────────────────────────────────────
def normalize_and_draw(white, all_pts, img_size=400, padding=30):
    """
    ✅ SAME normalization as convert_to_skeleton.py
    Normalizes ALL hand landmarks together to fill canvas.
    Fixes tiny/misplaced skeletons regardless of hand distance from camera.
    """
    all_x = []
    all_y = []
    for pts in all_pts:
        for p in pts:
            all_x.append(p[0])
            all_y.append(p[1])

    if not all_x:
        return False

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    range_x = max_x - min_x
    range_y = max_y - min_y

    if range_x == 0: range_x = 1
    if range_y == 0: range_y = 1

    scale_x = (img_size - 2 * padding) / range_x
    scale_y = (img_size - 2 * padding) / range_y
    scale   = min(scale_x, scale_y)

    scaled_w = range_x * scale
    scaled_h = range_y * scale
    offset_x = (img_size - scaled_w) / 2
    offset_y = (img_size - scaled_h) / 2

    colors = [(0, 200, 0), (255, 100, 0)]   # GREEN, BLUE

    for hand_idx, pts in enumerate(all_pts):
        color = colors[hand_idx % 2]

        norm_pts = []
        for p in pts:
            nx = int((p[0] - min_x) * scale + offset_x)
            ny = int((p[1] - min_y) * scale + offset_y)
            norm_pts.append((nx, ny))

        def draw_line(a, b):
            cv2.line(white, norm_pts[a], norm_pts[b], color, 3)

        for t in range(0, 4):   draw_line(t, t + 1)
        for t in range(5, 8):   draw_line(t, t + 1)
        for t in range(9, 12):  draw_line(t, t + 1)
        for t in range(13, 16): draw_line(t, t + 1)
        for t in range(17, 20): draw_line(t, t + 1)
        draw_line(5, 9);   draw_line(9, 13)
        draw_line(13, 17); draw_line(0, 5); draw_line(0, 17)

        for pt in norm_pts:
            cv2.circle(white, pt, 3, (0, 0, 255), -1)

    return True


# ─── APPLICATION ─────────────────────────────────────────────────────────────
class Application:

    def __init__(self):
        self.vs = cv2.VideoCapture(0)
        self.current_image  = None
        self.current_image2 = None

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at: {MODEL_PATH}\n"
                "Please run train_isl_model.py first."
            )
        self.model = load_model(MODEL_PATH)
        print("✅ ISL Model loaded successfully!")
        print(f"   Input shape  : {self.model.input_shape}")
        print(f"   CTC buffer   : {CTC_BUFFER_SIZE} frames")
        print(f"   Blank thresh : {CONFIDENCE_THRESHOLD}")

        # TTS
        self.speak_engine = pyttsx3.init()
        self.speak_engine.setProperty("rate", 100)
        voices = self.speak_engine.getProperty("voices")
        self.speak_engine.setProperty("voice", voices[0].id)

        # State
        self.ct             = {'blank': 0}
        self.prev_char      = ""
        self.count          = -1
        self.ten_prev_char  = [" "] * 10
        self.pts            = [(0, 0)] * 21
        self.pts2           = [(0, 0)] * 21
        self.hands_count    = 0

        for letter in ascii_uppercase:
            self.ct[letter] = 0

        self.str            = " "
        self.word           = " "
        self.word1          = " "
        self.word2          = " "
        self.word3          = " "
        self.word4          = " "
        self.current_symbol = "..."

        # CTC buffers
        self.ctc_raw_buffer  = []
        self.ctc_display     = ""
        self.cooldown_until  = 0
        self.confirmed_letters = []   # track all confirmed letters

        # ── GUI ──────────────────────────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("ISL — Indian Sign Language To Text Conversion")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1300x760")
        self.root.resizable(False, False)
        self.root.configure(bg="#f8f9fa")

        # ── Title bar ────────────────────────────────────────────────────────
        title_bar = tk.Frame(self.root, bg="#4263eb", height=50)
        title_bar.place(x=0, y=0, width=1300, height=50)
        tk.Label(title_bar,
                 text="Indian Sign Language To Text Conversion",
                 font=("Arial", 18, "bold"),
                 bg="#4263eb", fg="white").place(x=20, y=10)

        # ── Camera panel ─────────────────────────────────────────────────────
        cam_frame = tk.Frame(self.root, bg="#e9ecef",
                             relief="flat", bd=0)
        cam_frame.place(x=20, y=65, width=490, height=420)
        tk.Label(cam_frame, text="Live Camera",
                 font=("Arial", 10), bg="#e9ecef",
                 fg="#868e96").place(x=5, y=2)
        self.panel = tk.Label(cam_frame, bg="#dee2e6")
        self.panel.place(x=0, y=22, width=490, height=398)

        # ── Skeleton panel ───────────────────────────────────────────────────
        skel_frame = tk.Frame(self.root, bg="#e9ecef",
                              relief="flat", bd=0)
        skel_frame.place(x=525, y=65, width=420, height=420)
        tk.Label(skel_frame, text="Hand Skeleton",
                 font=("Arial", 10), bg="#e9ecef",
                 fg="#868e96").place(x=5, y=2)
        self.panel2 = tk.Label(skel_frame, bg="white",
                                relief="flat", borderwidth=0)
        self.panel2.place(x=0, y=22, width=420, height=398)

        # ── Character small window ────────────────────────────────────────────
        char_frame = tk.Frame(self.root, bg="#e7f5ff",
                              relief="flat", bd=0,
                              highlightbackground="#74c0fc",
                              highlightthickness=1)
        char_frame.place(x=960, y=65, width=320, height=200)
        tk.Label(char_frame, text="Character",
                 font=("Arial", 10), bg="#e7f5ff",
                 fg="#1971c2").place(x=10, y=8)
        self.panel3 = tk.Label(char_frame, text="...",
                                font=("Arial", 72, "bold"),
                                bg="#e7f5ff", fg="#1971c2")
        self.panel3.place(x=0, y=30, width=320, height=120)

        # Confidence bar canvas
        tk.Label(char_frame, text="Confidence",
                 font=("Arial", 9), bg="#e7f5ff",
                 fg="#868e96").place(x=10, y=158)
        self.conf_canvas = tk.Canvas(char_frame, bg="#e7f5ff",
                                     height=12, width=220,
                                     highlightthickness=0)
        self.conf_canvas.place(x=90, y=160)
        self.conf_canvas.create_rectangle(0, 0, 220, 12,
                                          fill="#dee2e6", outline="")
        self.conf_bar = self.conf_canvas.create_rectangle(0, 0, 0, 12,
                                                           fill="#4263eb",
                                                           outline="")

        # ── Hands + CTC info ─────────────────────────────────────────────────
        info_frame = tk.Frame(self.root, bg="#f8f9fa")
        info_frame.place(x=960, y=275, width=320, height=210)

        tk.Label(info_frame, text="Hands Detected",
                 font=("Arial", 10), bg="#f8f9fa",
                 fg="#868e96").place(x=0, y=5)
        self.panel_hands = tk.Label(info_frame, text="0",
                                     font=("Arial", 28, "bold"),
                                     bg="#f8f9fa", fg="#e03131")
        self.panel_hands.place(x=0, y=25)

        tk.Label(info_frame, text="CTC Decoded",
                 font=("Arial", 10), bg="#f8f9fa",
                 fg="#868e96").place(x=0, y=80)
        self.panel_ctc = tk.Label(info_frame, text="...",
                                   font=("Arial", 22, "bold"),
                                   bg="#f8f9fa", fg="#2f9e44")
        self.panel_ctc.place(x=0, y=100)

        # ── Divider ───────────────────────────────────────────────────────────
        div = tk.Frame(self.root, bg="#dee2e6", height=1)
        div.place(x=0, y=497, width=1300, height=1)

        # ── Sentence ──────────────────────────────────────────────────────────
        sent_frame = tk.Frame(self.root, bg="#ffffff",
                              highlightbackground="#dee2e6",
                              highlightthickness=1)
        sent_frame.place(x=20, y=508, width=1260, height=50)
        tk.Label(sent_frame, text="Sentence :",
                 font=("Arial", 13, "bold"),
                 bg="#ffffff", fg="#343a40").place(x=10, y=12)
        self.panel5 = tk.Label(sent_frame, text=" ",
                                font=("Arial", 14),
                                bg="#ffffff", fg="#1971c2",
                                wraplength=1050,
                                anchor="w", justify="left")
        self.panel5.place(x=120, y=12)

        # ── Suggestions ───────────────────────────────────────────────────────
        tk.Label(self.root, text="Suggestions :",
                 font=("Arial", 12, "bold"),
                 bg="#f8f9fa", fg="#e03131").place(x=20, y=575)

        btn_style = {"font": ("Arial", 12),
                     "bg": "#e7f5ff",
                     "fg": "#1971c2",
                     "relief": "flat",
                     "activebackground": "#74c0fc",
                     "activeforeground": "white",
                     "cursor": "hand2",
                     "bd": 1}

        self.b1 = tk.Button(self.root, **btn_style)
        self.b1.place(x=180, y=568, width=160, height=38)
        self.b2 = tk.Button(self.root, **btn_style)
        self.b2.place(x=350, y=568, width=160, height=38)
        self.b3 = tk.Button(self.root, **btn_style)
        self.b3.place(x=520, y=568, width=160, height=38)
        self.b4 = tk.Button(self.root, **btn_style)
        self.b4.place(x=690, y=568, width=160, height=38)

        # ── Speak / Clear / Backspace buttons ────────────────────────────────
        tk.Button(self.root, text="⌫ Backspace",
                  font=("Arial", 12),
                  bg="#fff3bf", fg="#e67700",
                  relief="flat", cursor="hand2",
                  activebackground="#ffd43b",
                  command=lambda: self._do_backspace()
                  ).place(x=900, y=568, width=120, height=38)

        tk.Button(self.root, text="Clear",
                  font=("Arial", 12, "bold"),
                  bg="#ffe3e3", fg="#c92a2a",
                  relief="flat", cursor="hand2",
                  activebackground="#ff6b6b",
                  activeforeground="white",
                  command=self.clear_fun
                  ).place(x=1035, y=568, width=100, height=38)

        tk.Button(self.root, text="Speak",
                  font=("Arial", 12, "bold"),
                  bg="#d3f9d8", fg="#2b8a3e",
                  relief="flat", cursor="hand2",
                  activebackground="#51cf66",
                  activeforeground="white",
                  command=self.speak_fun
                  ).place(x=1150, y=568, width=130, height=38)

        # ── Controls hint ─────────────────────────────────────────────────────
        tk.Label(self.root,
                 text="Hold gesture to fill bar → letter confirmed  |  Open palm = next  |  Peace ✌ = space  |  Thumbs up = backspace",
                 font=("Arial", 9),
                 bg="#f8f9fa", fg="#adb5bd").place(x=20, y=620)

        self.root.after(15, self.video_loop)

    # ─── VIDEO LOOP ──────────────────────────────────────────────────────────
    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            if not ok or frame is None:
                self.root.after(15, self.video_loop)
                return

            cv2image      = cv2.flip(frame, 1)
            cv2image_copy = np.array(cv2image)

            # Camera display
            rgb = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            # ✅ Detect ALL hands in full frame at once
            hands     = hd.findHands(cv2image, draw=False, flipType=True)
            hand_list = hands[0] if isinstance(hands, tuple) else hands

            white = np.ones((400, 400, 3), np.uint8) * 255

            if hand_list:
                self.hands_count = len(hand_list)
                self.panel_hands.config(
                    text=str(self.hands_count),
                    fg="#2f9e44" if self.hands_count == 1 else "#e67700"
                )

                # ✅ Collect landmarks from ALL hands directly from full frame
                all_pts = []
                for hand in hand_list:
                    pts_xy = [(p[0], p[1]) for p in hand['lmList']]
                    all_pts.append(pts_xy)

                # Store first hand pts for gesture detection
                self.pts  = all_pts[0]
                if len(all_pts) >= 2:
                    self.pts2 = all_pts[1]

                # ✅ Normalize ALL hands together — same as training!
                success = normalize_and_draw(white, all_pts, img_size=400)

                if success:
                    self.predict(white)

                    # Show skeleton
                    skeleton_rgb = cv2.cvtColor(white, cv2.COLOR_BGR2RGB)
                    self.current_image2 = Image.fromarray(skeleton_rgb)
                    imgtk2 = ImageTk.PhotoImage(image=self.current_image2)
                    self.panel2.imgtk = imgtk2
                    self.panel2.config(image=imgtk2)

                    self.panel3.config(text=str(self.current_symbol))
                    self.panel_ctc.config(text=self.ctc_display)
                    self.panel5.config(text=self.str)
                    self.b1.config(text=self.word1, command=self.action1)
                    self.b2.config(text=self.word2, command=self.action2)
                    self.b3.config(text=self.word3, command=self.action3)
                    self.b4.config(text=self.word4, command=self.action4)

            else:
                self.hands_count = 0
                self.panel_hands.config(text="0", fg="red")

        except Exception:
            print("== video_loop error ==")
            print(traceback.format_exc())
        finally:
            self.root.after(15, self.video_loop)

    # ─── PREDICT ─────────────────────────────────────────────────────────────
    def predict(self, skeleton_400):
        resized = cv2.resize(skeleton_400, (IMG_SIZE, IMG_SIZE))
        inp     = resized.reshape(1, IMG_SIZE, IMG_SIZE, 3).astype('float32') / 255.0

        prob       = np.array(self.model.predict(inp, verbose=0)[0], dtype='float32')
        ch1_idx    = int(np.argmax(prob))
        confidence = float(prob[ch1_idx])

        # Blank if low confidence
        if confidence < CONFIDENCE_THRESHOLD:
            raw_pred = BLANK_TOKEN
        else:
            raw_pred = LETTERS[ch1_idx]

        self.current_symbol = raw_pred
        self._update_conf_bar(confidence)

        # ── Cooldown check — ignore frames after letter confirmed ─────────────
        import time
        if hasattr(self, 'cooldown_until') and time.time() < self.cooldown_until:
            remaining  = self.cooldown_until - time.time()
            bar_filled = int((1 - remaining / COOLDOWN_SECONDS) * 10)
            bar        = "█" * bar_filled + "░" * (10 - bar_filled)
            self.panel3.config(text=f"Wait...  [{bar}]")
            return

        # ── CTC buffer ────────────────────────────────────────────────────────
        self.ctc_raw_buffer.append(raw_pred)

        # Show live progress bar
        progress   = len(self.ctc_raw_buffer)
        bar_filled = int((progress / CTC_BUFFER_SIZE) * 10)
        bar        = "█" * bar_filled + "░" * (10 - bar_filled)
        self.panel3.config(text=f"{raw_pred}  [{bar}]")

        if len(self.ctc_raw_buffer) >= CTC_BUFFER_SIZE:
            # CTC decode — remove consecutive duplicates and blanks
            decoded = []
            prev    = None
            for p in self.ctc_raw_buffer:
                if p != prev:
                    if p != BLANK_TOKEN:
                        decoded.append(p)
                    prev = p
            decoded_str = "".join(decoded)

            # Print in exact format like screenshot
            print(f"Raw buffer  : {self.ctc_raw_buffer}")
            print(f"CTC decoded : '{decoded_str}'  "
                  f"[{'2-hand' if self.hands_count == 2 else '1-hand'}]")
            print("-" * 50)

            if decoded_str:
                self.ctc_display    = decoded_str
                # Add only first decoded letter to sentence with cooldown
                first_letter        = decoded_str[0]
                self.str           += first_letter
                self.confirmed_letters.append(first_letter)
                self.cooldown_until = time.time() + COOLDOWN_SECONDS

            self.ctc_raw_buffer = []

        # Gesture overrides
        pts = self.pts

        # Thumb up = space
        if (pts[4][1] < pts[3][1] and
                pts[4][1] < pts[8][1] and
                pts[6][1]  < pts[8][1] and
                pts[10][1] < pts[12][1] and
                pts[14][1] < pts[16][1] and
                pts[18][1] < pts[20][1]):
            if self.prev_char != "next":
                self.str += " "
            self.current_symbol = "next"
            self.prev_char = "next"
            self._update_suggestions()
            return

        # Fist = backspace
        if (pts[0][0] > pts[8][0] and
                pts[0][0] > pts[12][0] and
                pts[0][0] > pts[16][0] and
                pts[0][0] > pts[20][0] and
                pts[4][1] < pts[8][1] and
                pts[4][1] < pts[12][1] and
                pts[4][1] < pts[16][1] and
                pts[4][1] < pts[20][1]):
            if self.prev_char != "Backspace":
                if len(self.str) > 1:
                    self.str = self.str[:-1]
            self.current_symbol = "Backspace"
            self.prev_char = "Backspace"
            self._update_suggestions()
            return

        self.prev_char = raw_pred
        self.count    += 1
        self.ten_prev_char[self.count % 10] = raw_pred
        self._update_suggestions()

    # ─── SUGGESTIONS ─────────────────────────────────────────────────────────
    def _update_suggestions(self):
        self.word1 = self.word2 = self.word3 = self.word4 = " "
        try:
            stripped = self.str.strip()
            if not stripped:
                return
            st   = self.str.rfind(" ")
            word = self.str[st + 1:]
            self.word = word
            if len(word.strip()) < 1:
                return
            suggestions = ddd.suggest(word)
            if len(suggestions) >= 1: self.word1 = suggestions[0]
            if len(suggestions) >= 2: self.word2 = suggestions[1]
            if len(suggestions) >= 3: self.word3 = suggestions[2]
            if len(suggestions) >= 4: self.word4 = suggestions[3]
        except Exception:
            pass

    def distance(self, x, y):
        return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

    def _do_backspace(self):
        if len(self.str) > 1:
            self.str = self.str[:-1]
            self.panel5.config(text=self.str)

    def _update_conf_bar(self, confidence):
        width = int(confidence * 220)
        self.conf_canvas.coords(self.conf_bar, 0, 0, width, 12)
        color = "#2f9e44" if confidence > 0.8 else "#f59f00" if confidence > 0.6 else "#e03131"
        self.conf_canvas.itemconfig(self.conf_bar, fill=color)

    def _replace_current_word(self, replacement):
        if replacement.strip() == "":
            return
        st  = self.str.rfind(" ")
        idx = self.str.find(self.word, st)
        if idx != -1:
            self.str = self.str[:idx] + replacement.upper()
        else:
            self.str = self.str.rstrip() + " " + replacement.upper()
        self._update_suggestions()

    def action1(self): self._replace_current_word(self.word1)
    def action2(self): self._replace_current_word(self.word2)
    def action3(self): self._replace_current_word(self.word3)
    def action4(self): self._replace_current_word(self.word4)

    def speak_fun(self):
        text = self.str.strip()
        if text:
            self.speak_engine.say(text)
            self.speak_engine.runAndWait()

    def clear_fun(self):
        self.str             = " "
        self.word            = " "
        self.word1 = self.word2 = self.word3 = self.word4 = " "
        self.prev_char       = ""
        self.current_symbol  = "..."
        self.ctc_raw_buffer  = []
        self.ctc_display     = ""
        self.panel3.config(text="...")
        self.panel_ctc.config(text="...")
        self.panel5.config(text=" ")
        self.b1.config(text=" ")
        self.b2.config(text=" ")
        self.b3.config(text=" ")
        self.b4.config(text=" ")

    def destructor(self):
        print("Closing ISL application...")
        print(self.ten_prev_char)
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting ISL Application...")
    app = Application()
    app.root.mainloop()
