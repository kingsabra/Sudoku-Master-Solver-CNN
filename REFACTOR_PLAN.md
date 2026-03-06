# Sudoku Master Solver CNN — Repo Scan & Refactor Plan

## 1. Repo scan: what the project actually does

### High-level summary

The repo implements an **end-to-end pipeline** that:

1. **Takes a photo of a Sudoku grid** (e.g. from `images/`).
2. **Detects the grid and cells** via image preprocessing (grayscale, binarization, connected components, optional Hough lines).
3. **Recognizes digits in each cell** using a **small CNN trained with backpropagation on MNIST** (handwritten digits). The CNN does **digit recognition only** (1–9 or empty).
4. **Builds a 9×9 board** from the CNN’s predictions.
5. **Solves the puzzle** with a **classical recursive backtracking solver** (constraint propagation + DFS), **not** with a neural network.

So:

- **Backpropagation** = used to **train the CNN** (digit classifier) on MNIST.
- **Sudoku solving** = **deterministic backtracking** (`solveSudoku` → `solverec` → `getCandidates`). There is no “CNN that learns to solve Sudoku” or backprop for the solver.

The project is best described as: **“Sudoku grid recognition with a backprop-trained CNN + classical backtracking solver.”**

### Component map

| Component | Role |
|----------|------|
| `Test_Solver.m` | Main script: load images → ground truth (OCR) → train CNN on MNIST → run CNN on cells → compare to ground truth → call `solveSudoku` → show solution. |
| `MnistConv.m` | CNN training: SGD + momentum, backprop through Conv → ReLU → Pool → FC → ReLU → Softmax. |
| `Conv.m`, `ReLU.m`, `Pool.m`, `Softmax.m` | CNN building blocks (forward pass). |
| `LoadMINSTImg.m`, `LoadMINSTLabel.m` | Load MNIST images/labels from IDX format. |
| `solveSudoku.m`, `solverec.m`, `getCandidates.m` | Recursive backtracking Sudoku solver. |
| `addtional function/` | Helpers: `checkSudoku`, `sudokuEngine2` (intlinprog), `drawSudoku2`, `ImagePrep`, `ImagePrepMethod2`, `TestMnistConv`, `PlotFeatures`, `display_network`. |
| `rng.m` | Custom RNG seed (shadows MATLAB’s built-in `rng`). |
| `MnistConv.mat` | Saved trained weights (if generated). |

### Notable issues from the scan

- **Naming**: `MINST` → should be `MNIST`; folder `addtional function` → `additional_functions` or `utils`.
- **Bug**: `LoadMINSTLabel.m` uses `assert(fp ~= 1, ...)`; should be `fp ~= -1` for fopen failure.
- **Design**: CNN is **retrained inside the loop** in `Test_Solver.m` for every test image; training should be done once, then reuse `MnistConv.mat` for inference.
- **Portability**: Hardcoded `images\` and `\` path separators; `Final_acc = (sum(total_acc)/20)` assumes exactly 20 files.
- **Code quality**: Many typos (e.g. “teh”, “poistion”, “coulo”, “thersh”, “messege”), duplicated OCR→board logic, long monolithic script.
- **Git**: `MnistConv.mat`, `MNIST/` data, and `images/` are good candidates for `.gitignore` (large/binaries); README should document how to obtain MNIST and sample images.

---

## 2. Refactor plan (clean, advanced, professional, performant)

### Phase 1: Repo structure and naming

- **Directory layout** (suggested):
  ```
  Sudoku-Master-Solver-CNN/
  ├── README.md
  ├── REFACTOR_PLAN.md          # (this file; optional to keep after refactor)
  ├── LICENSE
  ├── .gitignore
  ├── docs/                     # optional: architecture diagram, design notes
  ├── src/
  │   ├── cnn/                  # CNN only
  │   │   ├── Conv.m
  │   │   ├── ReLU.m
  │   │   ├── Pool.m
  │   │   ├── Softmax.m
  │   │   ├── MnistConv.m       # train
  │   │   ├── LoadMNISTImg.m    # rename MINST → MNIST
  │   │   ├── LoadMNISTLabel.m
  │   │   └── rng_seed.m        # rename to avoid shadowing built-in rng
  │   ├── solver/               # backtracking only
  │   │   ├── solveSudoku.m
  │   │   ├── solverec.m
  │   │   └── getCandidates.m
  │   ├── vision/               # grid detection + cell extraction
  │   │   ├── preprocessSudokuImage.m
  │   │   ├── extractGrid.m
  │   │   └── getCellImages.m
  │   └── pipeline/
  │       ├── runPipeline.m     # single entry: image → board → solution
  │       ├── trainCNN.m        # one-off training, save MnistConv.mat
  │       └── recognizeBoard.m  # load model + run CNN on 81 cells
  ├── utils/                    # rename from "addtional function"
  │   ├── checkSudoku.m
  │   ├── drawSudoku.m          # rename drawSudoku2
  │   ├── sudokuEngine2.m       # optional BIP solver
  │   ├── ImagePrep.m
  │   ├── ImagePrepMethod2.m
  │   ├── display_network.m
  │   ├── PlotFeatures.m
  │   └── TestMnistConv.m
  ├── scripts/
  │   ├── train_model.m         # train once, save weights
  │   └── run_demo.m            # run on images/ with optional figures
  ├── data/                     # or keep MNIST at top level with .gitignore
  │   └── MNIST/                # t10k-images.idx3-ubyte, t10k-labels.idx1-ubyte
  ├── images/                   # sample Sudoku photos (gitignore or sample only)
  └── models/                   # MnistConv.mat (gitignore, document download/train)
  ```
- **Rename files**: `LoadMINSTImg` → `LoadMNISTImg`, `LoadMINSTLabel` → `LoadMNISTLabel`; fix all call sites.
- **Rename folder**: `addtional function` → `utils` (or `additional_functions`) and fix paths.

### Phase 2: Code quality and correctness

- **Fix LoadMNISTLabel**: use `assert(fp ~= -1, ...)` for file open check.
- **Fix rng**: rename to `rng_seed.m` (or `set_rng.m`) so it doesn’t shadow MATLAB’s `rng`; update callers.
- **Spelling/typos**: fix “colomns/colom”, “teh”, “poistion”, “thersh”, “messege”, “Cloud” → “Could”, “segementing”, etc., in comments and strings.
- **Remove duplicate logic**: Ground-truth board construction (OCR path) and CNN path share ideas; consider one shared “cell → digit” helper and separate “how we get the image of the cell” (OCR vs CNN).
- **Magic numbers**: replace literals (e.g. 180, 0.4, 34, 20 peaks, 8000 samples, 4 epochs) with named constants or a small config (e.g. `config.m` or `opts` struct).

### Phase 3: Performance and design

- **Train once, infer many**: Move CNN training out of the per-image loop. Add `scripts/train_model.m` that trains on MNIST and saves `MnistConv.mat`. `Test_Solver` (or `run_demo.m`) should load `MnistConv.mat` and only run inference per image.
- **Batch inference (optional)**: For the 81 cells, you can batch the forward pass (e.g. stack 81 crops into a 4D array and run Conv/Pool/FC in batch) to reduce loop overhead (requires careful reshaping in your Conv/Pool).
- **Solver**: Current backtracking + `getCandidates` is already efficient; optional: add simple heuristics (e.g. choose cell with fewest candidates first) to reduce branches.
- **Optional BIP solver**: Keep `sudokuEngine2` (intlinprog) as an alternative; document it as “optional MathWorks-style BIP solver” in README.

### Phase 4: Robustness and portability

- **Paths**: Use `fullfile('images', ...)`, `fullfile('MNIST', ...)` (or `data/MNIST`) so it works on Windows and Unix.
- **Final_acc**: Don’t assume 20 files; use `num_files` (e.g. `mean(total_acc)` or `sum(total_acc)/numel(total_acc)`).
- **Error handling**: Check `fopen`, existence of `MnistConv.mat` and MNIST files; clear messages (e.g. “Place MNIST files in data/MNIST/”).
- **Empty/edge cases**: Handle “no images found”, “no grid detected”, “unsolvable puzzle” with clear messages or flags.

### Phase 5: GitHub and documentation

- **.gitignore**: Add `*.mat`, `MNIST/`, `data/`, `images/*.jpg`, `images/*.png` (or list allowed sample images); keep `READ ME.pdf` out of repo or in `docs/` if needed.
- **LICENSE**: Add a license (e.g. MIT) if you want it open source.
- **README**: Single, clear README with:
  - **Title and one-line description**
  - **Architecture**: pipeline diagram (image → preprocessing → grid/cells → CNN digit recognition → 9×9 board → backtracking solver → solution).
  - **Use cases**: “Solve Sudoku from a photo”, “benchmark digit recognition on printed Sudoku”, “reuse solver on programmatic boards.”
  - **Functionality**: what each part does (CNN training vs inference, solver, optional BIP).
  - **Requirements**: MATLAB version, toolboxes (Image Processing, Optimization for intlinprog if used).
  - **Setup**: where to put MNIST files, how to run training once, how to run demo.
  - **Results**: section with placeholders for accuracy (e.g. digit recognition %, full-board accuracy %, example images/solutions). You can fill numbers and screenshots later.
- **Comments in code**: Short headers for each file (purpose, inputs/outputs); keep complex logic explained.

### Phase 6: Optional “advanced” touches

- **Config**: Single `config.m` or `opts.m` for paths, training hyperparameters, and inference options (e.g. whether to show figures).
- **Tests**: Simple script that runs `solveSudoku` on a few known 9×9 boards and checks the solution (no CNN needed).
- **Results script**: `scripts/evaluate.m` that runs the pipeline on `images/`, computes accuracy, and optionally saves a table or plot for the README.

---

## 3. README outline (for the new README)

Suggested sections (you can paste this into README and fill in):

1. **Project title** — e.g. “Sudoku Master Solver — CNN digit recognition + backtracking.”
2. **Brief description** — One paragraph: photo of Sudoku → CNN recognizes digits → backtracking solves the puzzle.
3. **Architecture** — Diagram or bullet flow: Image → Preprocessing → Grid/Cells → CNN (Conv–ReLU–Pool–FC–Softmax) → 9×9 board → Backtracking solver → Solution. Mention backprop for **training the CNN**, not for solving.
4. **Use cases** — (1) Solve from photo, (2) Benchmark recognition, (3) Use solver on given 9×9 matrix.
5. **Requirements** — MATLAB R20xx+, Image Processing Toolbox; Optimization Toolbox only if using BIP solver.
6. **Setup** — Clone repo; download MNIST (or use script); place test images in `images/`.
7. **Usage** — “Train once: `scripts/train_model`; run demo: `scripts/run_demo`” (or current entry point after refactor).
8. **Repository structure** — Short tree of main folders (`src/cnn`, `src/solver`, `src/vision`, `utils`, `scripts`, `data`, `images`).
9. **Results** — Placeholder: “Digit recognition accuracy: X%. Board accuracy: Y%. Example: [screenshot of input/output].” You can add real numbers and figures when ready.
10. **References** — MNIST, any papers or MathWorks examples you relied on.
11. **License** — e.g. MIT.

---

## 4. Clarification for presentation

When you present the project, it’s worth stating clearly:

- **Backpropagation** is used to **train the digit-recognition CNN** on MNIST.
- **Sudoku solving** is **classical backtracking** (no learning there).
- The “CNN” part is the **recognition stage**; the “solver” is **deterministic and exact**.

That keeps the narrative accurate and still highlights the ML component (CNN + backprop) and the full pipeline (image → digits → solution).

If you share your target MATLAB version and whether you want to keep the optional BIP solver and `READ ME.pdf`, the refactor can be adjusted accordingly. Once you have accuracy numbers and example images, we can fill the **Results** section in the README.
