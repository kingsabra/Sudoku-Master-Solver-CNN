# Sudoku Master Solver — CNN Digit Recognition + Backtracking

A MATLAB pipeline that **solves Sudoku from a photo**: a convolutional neural network (CNN) trained with **backpropagation** on MNIST recognizes digits in each cell, then a **recursive backtracking** solver finds the solution.

---

## Description

The project combines:

1. **Image preprocessing** — Detect the Sudoku grid, extract 81 cell images.
2. **Digit recognition** — A small CNN (Conv → ReLU → Pool → FC → Softmax) trained on MNIST with backpropagation classifies each cell as digit 1–9 or empty.
3. **Sudoku solving** — A deterministic backtracking solver (constraint propagation + DFS) solves the 9×9 puzzle.

**Note:** Backpropagation is used to **train the CNN** (digit classifier). The **solver** is classical recursion, not a neural network.

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Sudoku image   │ ──► │  Preprocessing    │ ──► │  81 cell crops  │
│  (e.g. photo)   │     │  (crop, resize)   │     │  (e.g. 30×30)   │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Solved 9×9     │ ◄── │  Backtracking     │ ◄── │  CNN digit      │
│  board          │     │  solver          │     │  recognition    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### CNN (digit recognition)

- **Training:** MNIST dataset; SGD with momentum; backprop through Conv → ReLU → Pool → FC → ReLU → Softmax.
- **Layers:** 9×9 conv (20 filters) → ReLU → 2×2 pool → flatten → FC(2000→100) → ReLU → FC(100→10) → Softmax.
- **Output:** Class 1–9 or empty (handled by a simple “ink” threshold per cell).

### Solver (backtracking)

- **Method:** Recursive DFS with candidate generation from row/column/3×3 constraints (`getCandidates`).
- **Entry:** `solveSudoku(board)` → `solverec`; board uses `NaN` for empty cells.

---

## Use Cases

| Use case | Description |
|----------|-------------|
| **Solve from photo** | Run the full pipeline on an image of a printed or displayed Sudoku grid. |
| **Digit recognition benchmark** | Compare CNN output to ground truth (e.g. OCR or manual) to report per-cell or full-board accuracy. |
| **Solver only** | Call `solveSudoku(board)` with a 9×9 matrix (digits 1–9, `NaN` for empty) to get a solution without any CNN or images. |

---

## Requirements

- **MATLAB** (R2016a or later recommended).
- **Image Processing Toolbox** (for preprocessing, Hough, etc.).
- **Optimization Toolbox** only if you use the optional BIP solver (`utils/sudokuEngine2.m`).

---

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Sudoku-Master-Solver-CNN.git
   cd Sudoku-Master-Solver-CNN
   ```

2. **MNIST data**  
   Place MNIST test (or train) files in the `MNIST/` folder:
   - `t10k-images.idx3-ubyte`
   - `t10k-labels.idx1-ubyte`  
   Download from [MNIST](http://yann.lecun.com/exdb/mnist/) or your preferred mirror.

3. **Test images**  
   Put Sudoku photos in the `images/` folder. The demo accepts **all** `.jpg`, `.jpeg`, and `.png` images.

---

## Usage

**From repository root** (e.g. `cd Sudoku-Master-Solver-CNN` in MATLAB):

### Quick start (base model only)

1. **Train the CNN once** (saves `models/MnistConv.mat`):
   ```matlab
   scripts/train_model
   ```

2. **Run the full pipeline**:
   ```matlab
   Test_Solver
   ```
   Or: `scripts/run_demo`  
   Uses images in `images/` (`.jpg/.jpeg/.png`), compares to OCR ground truth, and runs the backtracking solver.

### AI pipeline (learns from your Sudoku photos — Phases 1–5)

1. **One-shot setup** (synthetic data + optional images → dataset → fine-tune):
   ```matlab
   scripts/setup_ai_pipeline
   ```
   Requires MNIST in `MNIST/` for base training. Creates `models/MnistConv_sudoku.mat`.

2. **Or step by step:**
   ```matlab
   scripts/generate_synthetic_cells    % Phase 1: synthetic 28×28 digit cells
   scripts/build_dataset_from_images   % Phase 1: extract cells from images/ (if any)
   scripts/train_model                 % Base model (if not done)
   scripts/train_sudoku_cells          % Phase 2: fine-tune on Sudoku cells
   ```

3. **Run pipeline** — `Test_Solver` or `scripts/run_demo` now prefer `MnistConv_sudoku.mat` and:
   - Use **batch inference** (Phase 4) for speed
   - **Confidence fallback** to OCR when CNN is uncertain
   - **Append (cell, solution)** to `data/sudoku_cells/` after each image (Phase 3), so the dataset grows over time

4. **Re-train periodically** (to actually “learn from itself”): after you’ve run the demo on more images and the dataset has grown, run:
   ```matlab
   scripts/train_sudoku_cells
   ```
   This refreshes `models/MnistConv_sudoku.mat` using the accumulated dataset.

4. **End-to-end from one image** (Phase 5):
   ```matlab
   [board, solution] = image_to_solution(pwd, 'images/your_sudoku.jpg');
   ```

### Solver only (no images, no CNN). Add `src/solver` to path first:
   ```matlab
   addpath('src/solver');
   board = [5 3 NaN NaN 7 NaN NaN NaN NaN;
            6 NaN NaN 1 9 5 NaN NaN NaN;
            NaN 9 8 NaN NaN NaN NaN 6 NaN;
            8 NaN NaN NaN 6 NaN NaN NaN 3;
            4 NaN NaN 8 NaN 3 NaN NaN 1;
            7 NaN NaN NaN 2 NaN NaN NaN 6;
            NaN 6 NaN NaN NaN NaN 2 8 NaN;
            NaN NaN NaN 4 1 9 NaN NaN 5;
            NaN NaN NaN NaN 8 NaN NaN 7 9];
   solved = solveSudoku(board);
   ```

---

## Repository structure

```
Sudoku-Master-Solver-CNN/
├── README.md
├── REFACTOR_PLAN.md          # Refactor notes (optional)
├── FOLLOW_UP_PLAN.md         # Roadmap: learned model from Sudoku photos
├── .gitignore
├── Test_Solver.m             # Entry point: runs scripts/run_demo
├── src/
│   ├── cnn/                  # CNN (backprop training + inference)
│   │   ├── Conv.m, ReLU.m, Pool.m, Softmax.m, CNNForwardBatch.m
│   │   ├── MnistConv.m, MnistConvFineTune.m, LoadMNISTImg.m, LoadMNISTLabel.m, rng_seed.m
│   ├── solver/               # Backtracking solver
│   │   ├── solveSudoku.m, solverec.m, getCandidates.m
│   └── vision/               # Grid detection (Phase 5)
│       └── detect_grid.m
├── scripts/
│   ├── train_model.m         # Base model → models/MnistConv.mat
│   ├── train_sudoku_cells.m  # Phase 2: fine-tune → models/MnistConv_sudoku.mat
│   ├── run_demo.m            # Full pipeline (batch inference, confidence, append to dataset)
│   ├── build_dataset_from_images.m   # Phase 1: extract cells from photos
│   ├── generate_synthetic_cells.m     # Phase 1: synthetic digit cells
│   ├── setup_ai_pipeline.m   # One-shot setup for AI pipeline
│   └── image_to_solution.m   # Phase 5: end-to-end image → solution
├── utils/                   # Helpers (checkSudoku, drawSudoku, etc.)
├── models/                   # MnistConv.mat, MnistConv_sudoku.mat
├── data/sudoku_cells/        # Sudoku cell dataset (Phase 1–3)
├── MNIST/                    # MNIST files (user-provided)
└── images/                   # Sudoku photos (.jpg/.jpeg/.png)
```

---

## Results

*Fill in with your own numbers and figures when ready.*

| Metric | Value |
|--------|--------|
| **Digit recognition accuracy** | *e.g. XX% on test cells* |
| **Full-board accuracy (81/81 correct)** | *e.g. XX% of boards* |
| **Solver** | Exact; finds solution when puzzle is valid and solvable. |

**Example:**  
*Add a screenshot or two: input Sudoku image → recognized board → solved board.*

---

## References

- [MNIST database](http://yann.lecun.com/exdb/mnist/)
- MathWorks examples (e.g. Sudoku via `intlinprog`) for the optional `sudokuEngine2` BIP solver.

---

## License

*Add your license here (e.g. MIT).*
