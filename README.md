# AI651 Assignment 2: Deep Learning for Space, Time and Graphs

- **Question 1** (River Forecasting Proposal) is a written design. Its text is in `report/q1.html` and the architecture diagram is drawn by `Question1/design/architecture_diagram.py`. `Question1/Question1.ipynb` is the optional companion notebook supplied with the assignment.
- **Question 2** (Motion or Memory?) contains all code, results, figures and GIFs, under `Question2/`.
- The submitted report is built from `report/` into `report/report.pdf`.

## Environment

Python 3.10, PyTorch 2.10, torchvision 0.25, OpenCV (`opencv-python-headless`), matplotlib, numpy and Pillow.
Experiments ran on an Apple M3 Pro, using the PyTorch MPS backend.

```bash
python -m venv .venv && .venv/bin/pip install torch torchvision opencv-python-headless matplotlib numpy pillow
```

The Moving MNIST file (`mnist_test_seq.npy`, 820 MB) is downloaded by
`torchvision.datasets.MovingMNIST(root="Question2/data", split=None, download=True)` the first time it is needed.

## Question 2 layout

| Path | Contents |
|------|----------|
| `Question2/mm/data.py` | Loading, the video split (0–7999 / 8000–8999 / 9000–9999), mirror/rotation and the group G, windows at speed k |
| `Question2/mm/models.py` | ConvLSTM cell (no peepholes), model A, the model-D candidates, the symmetrised wrapper |
| `Question2/mm/train.py` | Shared training protocol (Adam 1e-3, batch 32, 30 epochs, seeds 0/1, speed and symmetry augmentation) |
| `Question2/mm/evaluate.py` | Rollout prediction, MSE_h / E(5) / E(2), consistency gap C_T and transformed error E_T, inference timing |
| `Question2/scripts/` | One script per table/figure group (see below) |
| `Question2/checkpoints/` | Trained weights and per-epoch training logs (`<model>_s<seed>.pt/.json`) |
| `Question2/results/` | Every number reported, as JSON, plus per-video consistency gaps |
| `Question2/figures/`, `Question2/gifs/` | Figures 1–6 and the animated GIFs of Figures 3 and 6(a) |
| `Question2/predictions/` | Q2.5 and Q2.11 predictions, each committed before its experiments ran |

## Reproducing Question 2 (run from `Question2/`)

```bash
PY=../.venv/bin/python
$PY -m scripts.table1_checks          # Table 1
$PY -m scripts.figure1_flow           # Figure 1
./scripts/run_phase1_modelA.sh        # train model A (seeds 0, 1)
$PY -m scripts.baseline_table2        # Table 2, Figure 2
$PY -m scripts.test_baseline          # Tables 3-4, Figures 3-5 (+ GIFs)
./scripts/run_phase2_BCD.sh           # models B, C, D candidates (validation choice), D, D+B
$PY -m scripts.final_tables --final DB --with-db   # Tables 5-6, Figure 6 (+ GIFs)
cd ../report && ../.venv/bin/python build.py       # report.html -> report.pdf
```
