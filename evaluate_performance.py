import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from jiwer import wer
import random

# ─── PATHS ───────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "isl_model.h5")
DATASET_DIR   = r"C:\Users\vaayu\Downloads\ISL PROJECT\ISL_skeleton_data"

IMG_SIZE      = 64
VALID_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
idx_to_letter = {idx: letter for idx, letter in enumerate(VALID_LETTERS)}
IMAGES_PER_LETTER = 50

# ─── LOAD MODEL ──────────────────────────────────────────────────────────────
print("Loading model...")
model = load_model(MODEL_PATH)
print(f"Model loaded! Input shape: {model.input_shape}")

# ─── COLLECT PREDICTIONS ─────────────────────────────────────────────────────
print("\nTesting model on dataset images...")
print("=" * 60)

letter_references = []
letter_hypotheses = []
seq_references    = []
seq_hypotheses    = []

for letter in VALID_LETTERS:
    folder = os.path.join(DATASET_DIR, letter)
    if not os.path.exists(folder):
        continue

    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(files) == 0:
        continue

    sample_files = random.sample(files, min(IMAGES_PER_LETTER, len(files)))
    correct      = 0
    letter_preds = []

    for fname in sample_files:
        img = cv2.imread(os.path.join(folder, fname))
        if img is None:
            continue
        img  = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img  = img.reshape(1, IMG_SIZE, IMG_SIZE, 3).astype('float32') / 255.0
        prob = model.predict(img, verbose=0)[0]
        pred = idx_to_letter[int(np.argmax(prob))]

        letter_references.append(letter)
        letter_hypotheses.append(pred)
        letter_preds.append(pred)
        if pred == letter:
            correct += 1

    accuracy = (correct / len(sample_files)) * 100
    print(f"  {letter}: {correct}/{len(sample_files)} correct ({accuracy:.1f}%)")

    # Build 4-gram sequences for BLEU-4
    for i in range(0, len(letter_preds) - 3, 4):
        seq_references.append([letter] * 4)
        seq_hypotheses.append(letter_preds[i:i+4])

print("=" * 60)
print(f"Total letter samples    : {len(letter_references)}")
print(f"Total 4-gram sequences  : {len(seq_references)}")

# ─── LETTER ACCURACY ─────────────────────────────────────────────────────────
total_correct   = sum(1 for r, h in zip(letter_references, letter_hypotheses) if r == h)
letter_accuracy = (total_correct / len(letter_references)) * 100
print(f"\nOverall Letter Accuracy : {letter_accuracy:.2f}%")

# ─── BLEU SCORES ─────────────────────────────────────────────────────────────
print("\nComputing BLEU scores...")
smoother = SmoothingFunction().method4   # best smoothing for short sequences

# BLEU-1 on individual letters
bleu1_scores = []
for ref, hyp in zip(letter_references, letter_hypotheses):
    b1 = sentence_bleu([[ref]], [hyp],
                       weights=(1, 0, 0, 0),
                       smoothing_function=smoother)
    bleu1_scores.append(b1)
avg_bleu1 = np.mean(bleu1_scores)

# BLEU-4 on 4-gram sequences
bleu4_scores = []
for ref_seq, hyp_seq in zip(seq_references, seq_hypotheses):
    b4 = sentence_bleu([ref_seq], hyp_seq,
                       weights=(0.25, 0.25, 0.25, 0.25),
                       smoothing_function=smoother)
    bleu4_scores.append(b4)
avg_bleu4 = np.mean(bleu4_scores)

# Corpus BLEU
corpus_bleu1 = corpus_bleu([[[r]] for r in letter_references],
                            [[h] for h in letter_hypotheses],
                            weights=(1, 0, 0, 0),
                            smoothing_function=smoother)

corpus_bleu4 = corpus_bleu([[r] for r in seq_references],
                            seq_hypotheses,
                            weights=(0.25, 0.25, 0.25, 0.25),
                            smoothing_function=smoother)

# ─── WER ─────────────────────────────────────────────────────────────────────
print("Computing WER...")
wer_score = wer(" ".join(letter_references), " ".join(letter_hypotheses))

# ─── FINAL RESULTS ───────────────────────────────────────────────────────────
print("\n")
print("=" * 60)
print("         PERFORMANCE EVALUATION RESULTS")
print("=" * 60)
print(f"  Total Test Samples      : {len(letter_references)}")
print(f"  Letters Tested          : {len(VALID_LETTERS)} (A-Z)")
print(f"  Images Per Letter       : {IMAGES_PER_LETTER}")
print("-" * 60)
print(f"  Overall Letter Accuracy : {letter_accuracy:.2f}%")
print("-" * 60)
print(f"  BLEU-1 Score            : {avg_bleu1:.4f}  ({avg_bleu1*100:.2f}%)")
print(f"  BLEU-4 Score            : {avg_bleu4:.4f}  ({avg_bleu4*100:.2f}%)")
print(f"  Corpus BLEU-1           : {corpus_bleu1:.4f}  ({corpus_bleu1*100:.2f}%)")
print(f"  Corpus BLEU-4           : {corpus_bleu4:.4f}  ({corpus_bleu4*100:.2f}%)")
print("-" * 60)
print(f"  WER (Word Error Rate)   : {wer_score:.4f}  ({wer_score*100:.2f}%)")
print("=" * 60)
print("\nINTERPRETATION:")
print(f"  BLEU-1 > 0.80 is good  → Your score: {avg_bleu1:.4f} {'✅ GOOD' if avg_bleu1 > 0.80 else '❌ NEEDS IMPROVEMENT'}")
print(f"  BLEU-4 > 0.50 is good  → Your score: {avg_bleu4:.4f} {'✅ GOOD' if avg_bleu4 > 0.50 else '❌ NEEDS IMPROVEMENT'}")
print(f"  WER < 0.20 is good     → Your score: {wer_score:.4f} {'✅ GOOD' if wer_score < 0.20 else '❌ NEEDS IMPROVEMENT'}")
print("=" * 60)

# ─── SAVE RESULTS ────────────────────────────────────────────────────────────
results_path = os.path.join(BASE_DIR, "performance_results.txt")
with open(results_path, "w") as f:
    f.write("PERFORMANCE EVALUATION RESULTS\n")
    f.write("=" * 60 + "\n")
    f.write(f"Total Test Samples      : {len(letter_references)}\n")
    f.write(f"Letters Tested          : {len(VALID_LETTERS)} (A-Z)\n")
    f.write(f"Images Per Letter       : {IMAGES_PER_LETTER}\n")
    f.write("-" * 60 + "\n")
    f.write(f"Overall Letter Accuracy : {letter_accuracy:.2f}%\n")
    f.write("-" * 60 + "\n")
    f.write(f"BLEU-1 Score            : {avg_bleu1:.4f}  ({avg_bleu1*100:.2f}%)\n")
    f.write(f"BLEU-4 Score            : {avg_bleu4:.4f}  ({avg_bleu4*100:.2f}%)\n")
    f.write(f"Corpus BLEU-1           : {corpus_bleu1:.4f}  ({corpus_bleu1*100:.2f}%)\n")
    f.write(f"Corpus BLEU-4           : {corpus_bleu4:.4f}  ({corpus_bleu4*100:.2f}%)\n")
    f.write("-" * 60 + "\n")
    f.write(f"WER (Word Error Rate)   : {wer_score:.4f}  ({wer_score*100:.2f}%)\n")
    f.write("=" * 60 + "\n")

print(f"\nResults saved to: {results_path}")
print("Done!")
