# Follow-Up Plan: Toward an AI Model That Learns From Sudoku Photos

This document outlines how to evolve the current pipeline into a **model that learns from previously seen Sudoku photos** and becomes more accurate and performant over time, while **keeping the backtracking solver as-is** (no replacement with backprop for solving).

---

## Current Limitation

- The CNN is trained only on **MNIST** (handwritten digits). Sudoku photos often have **printed or screen-rendered digits**, different fonts, lighting, and perspective.
- The model does **not** learn from the Sudoku images you actually run: it never updates on (cell image → correct digit) pairs from your own data.

---

## Goal

An **artificially intelligent** pipeline that:

1. **Learns from past Sudoku photos** — Uses your own labeled or self-labeled data (cell crops + correct digit) to improve the digit classifier.
2. **Gets more performant** — Better accuracy on real Sudoku images; optionally faster inference (e.g. batch).
3. **Stays as-is for solving** — The Sudoku solution is still produced by the **backtracking solver**; only the **recognition** stage becomes learned and data-driven.

---

## Phase 1: Sudoku-Specific Dataset (Required for “Learning From Photos”)

**Idea:** Collect or generate **(cell image, digit)** pairs from real Sudoku photos and use them for training or fine-tuning.

1. **Data collection**
   - For each processed Sudoku image, you already have:
     - 81 cell crops (from the grid).
     - A **ground-truth board** (e.g. from OCR or from the known solution).
   - Save **cell crops** with **labels** (1–9 or “empty”) into a small dataset (e.g. `data/sudoku_cells/` with subfolders `1/`, `2/`, …, `9/`, `empty/`).
   - Optionally **generate** synthetic cells: render digits (e.g. in MATLAB) onto 30×30 patches and add noise/affine transforms to augment.

2. **Labeling**
   - **Self-labeling:** Use the current pipeline: run OCR (or the solver’s solution) to get the “true” board, then assign each cell crop the corresponding digit. No manual labeling required, but labels can be noisy.
   - **Semi-manual:** Manually correct a subset of boards; use them as gold labels for evaluation and fine-tuning.

3. **Format**
   - Store as **images per class** (e.g. `data/sudoku_cells/5/img_001.png`) or a single **MAT file** (e.g. `cellImages`: 30×30×N, `labels`: N×1). Document the format in the repo.

**Outcome:** A dataset of Sudoku cell images with labels, ready for training/fine-tuning.

---

## Phase 2: Fine-Tune the CNN on Sudoku Cells (Backpropagation on Your Data)

**Idea:** Keep the same **architecture** (Conv → ReLU → Pool → FC → Softmax) and **reuse** the current weights as initialization. Then train (or fine-tune) with **backpropagation** on your Sudoku cell dataset.

1. **Data pipeline**
   - Load your **(cell image, label)** pairs.
   - Resize/normalize to the size the CNN expects (e.g. 28×28 to match MNIST, or keep 30×30 and add a small adapter layer / resize in code).
   - Split into train / validation (e.g. 80/20).

2. **Training loop**
   - **Option A — Fine-tune:** Load `MnistConv.mat`, then run a few more epochs of `MnistConv` (or a copy that reads your data) with a **small learning rate** (e.g. `alpha = 0.001`) so the model adapts to Sudoku cells without forgetting MNIST entirely.
   - **Option B — Train from scratch on Sudoku only:** Initialize weights as you do now, train only on Sudoku cell data. Good if you have a large enough dataset.

3. **Implementation**
   - Add a script, e.g. `scripts/train_sudoku_cells.m`, that:
     - Loads `models/MnistConv.mat` (for Option A) or initializes new weights.
     - Loads your Sudoku cell dataset.
     - Runs backpropagation (same `MnistConv` or a variant that takes your data).
     - Saves updated weights to e.g. `models/MnistConv_sudoku.mat` and uses this in `run_demo` when available.

4. **Evaluation**
   - Report **accuracy on held-out Sudoku cells** and **full-board accuracy** (81/81) on a few test images. Compare to the MNIST-only model.

**Outcome:** A CNN that has **learned from Sudoku photos** (or from generated Sudoku-like cells), improving recognition on your target domain.

---

## Phase 3: Continuous Learning From New Photos (Optional)

**Idea:** Whenever you run the pipeline on a new image and have a **trusted** label (e.g. user confirms the solution, or you use the backtracking solution as proxy for “correct” clues), add those (cell, digit) pairs to the dataset and **periodically re-run fine-tuning**.

1. **Auto-labeling**
   - After solving, you have **solution board** and **detected board**. For cells where the solver filled a value, you can treat the solution as the “true” digit for that cell and add the corresponding crop + label to the dataset (with care: only for cells that were empty in the input and filled by the solver).
   - For **given clues**, the detected value is the “label” (or use OCR as ground truth when available).

2. **Retraining**
   - Every N new images (or weekly), run `train_sudoku_cells.m` on the accumulated dataset (with a train/val split) and replace `MnistConv_sudoku.mat`. Then `run_demo` uses this model.

**Outcome:** The system **improves over time** as more Sudoku photos are processed, without changing the solver.

---

## Phase 4: Performance and Robustness (Optional)

- **Batch inference:** Run the CNN on all 81 cells in one or a few batch forward passes (reshape cell stack to the format your Conv/Pool accept) to reduce per-image time.
- **Confidence / rejection:** Use Softmax probability to mark low-confidence cells and optionally fall back to OCR or ask the user.
- **Data augmentation:** When building the Sudoku cell dataset, augment with rotation, brightness, and blur so the model generalizes better.

---

## Phase 5: Future Directions (Beyond “Recognition + Backtracking”)

These go beyond “learn from photos for recognition” but are natural next steps if you want the **overall system** to feel more “AI”:

- **Learned grid detection:** Replace or assist hand-crafted preprocessing (Hough, etc.) with a small CNN or keypoint detector trained on Sudoku grid images.
- **End-to-end pipeline:** One network that takes the full image and outputs a 9×9 board (or 81 digit logits). Still use the **backtracking solver** on that output to guarantee a valid solution; the network only does recognition.
- **No learned solver:** Keep the backtracking solver as the only solver. Backprop is used only for the **recognition** part; the “intelligence” is in adapting to your photos, not in replacing the exact solver.

---

## Summary Table

| Phase | What | Backprop used for | Solver |
|-------|------|-------------------|--------|
| 1 | Build Sudoku cell dataset from photos (and/or synthetic) | — | Unchanged |
| 2 | Fine-tune or train CNN on Sudoku cells | Digit classifier | Unchanged |
| 3 | Add new photos to dataset; periodically re-train | Digit classifier | Unchanged |
| 4 | Batch inference, confidence, augmentation | — | Unchanged |
| 5 | Optional: learned grid detection, end-to-end board prediction | Recognition only | Still backtracking |

Implementing **Phases 1 and 2** gives you an **actual model that learns from Sudoku photos** via backpropagation on your data, while the Sudoku solution remains from the **backtracking** solver.
