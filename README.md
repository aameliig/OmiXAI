# OmiXAI

**OmiXAI** is an ensemble pipeline for gradient-based feature attribution in deep learning models trained on genomic and epigenomic data. It combines multiple attribution methods across CNN and GNN architectures and aggregates their outputs into a single ranked feature list via hybrid ranking.

Preprint: [bioRxiv 2025.04.28.651097](https://doi.org/10.1101/2025.04.28.651097)

---

## How it works

```mermaid
flowchart LR
    A["Genomic Data\nDNA + omics tracks"] --> B["Preprocessing\nOmicsDC · SparseVector\none-hot encoding"]
    B --> C["Train / Test Split\nStratifiedShuffleSplit\nby class + chromosome"]

    subgraph Training ["Model Training"]
        C --> D["ConvMZC\n12 × Conv2d\nF1=0.88 · AUC=0.97"]
        C --> E["GraphMZC\n13 × SAGEConv\nF1=0.81 · AUC=0.95"]
    end

    subgraph XAI ["XAI Interpretation — Captum + PyG"]
        D --> F["IG · IxG\nGB · Deconv"]
        E --> G["IG · IxG · GB\nDeconv · Saliency"]
        E --> H["GNNExplainer"]
    end

    F & G & H --> I["10 importance vectors\nper True Positive region"]
    I --> J["Hybrid Ranking\n% deviation from mean\naveraged across methods"]
    J --> K["Ranked feature list"]
    K --> L["Retrain with top-k\nvalidate F1 / AUC"]
```

---

## Supported attribution methods

| Method | CNN | GNN | Library |
|--------|:---:|:---:|---------|
| Integrated Gradients (IG) | ✓ | ✓ | Captum |
| InputXGradient (IxG) | ✓ | ✓ | Captum |
| Guided Backpropagation (GB) | ✓ | ✓ | Captum |
| Deconvolution (Deconv) | ✓ | ✓ | Captum |
| Saliency | — | ✓ | Captum |
| GNNExplainer | — | ✓ | PyG |

---

## Data

Genomic data: [vladislareon/z_dna](https://github.com/vladislareon/z_dna)

Feature serialisation uses [SparseVector](https://github.com/Nazar1997/Sparse_vector) — clone it and add to `PYTHONPATH` (see cluster setup below).

---

## Installation

### Developers (any OS, CPU or GPU)

One command — everything comes from PyPI:

```bash
git clone https://github.com/aameliig/OmiXAI.git && cd OmiXAI
pip install -e .            # or: pip install -r requirements.txt
```

`torch-scatter`/`torch-sparse`/`torch-cluster` are **not** required: modern
PyTorch Geometric (≥2.3) provides native fallbacks for the layers OmiXAI uses.
After `pip install -e .` the package is importable anywhere (`import omixai`)
and the runner scripts work from any directory.

### Exact GPU reproduction of the paper (CUDA 12.1)

Only if you want the precise pinned build used for the paper:

```bash
pip install -e ".[gpu]" \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
```

(torch must be present before the scatter/sparse wheels build, which is why this
path is pinned and index-specific.)

### HSE HPC cluster

Nothing to pip-install — the module ships torch 2.1.2+cu121, torch_geometric and
captum:

```bash
module load Python/Google_Colab_GPU_2024
git clone -b reviewer-revision https://github.com/aameliig/OmiXAI.git ~/OmiXAI
# SparseVector is not a pip package — clone alongside the data if not present:
git clone https://github.com/Nazar1997/Sparse_vector.git ~/DNA/Sparse_vector
```

---

## Running on a SLURM cluster

```bash
cd ~/OmiXAI

# Step 1 — interpretation (~4–8 h, 1 GPU). Default METHOD=hybrid.
sbatch scripts/omixai.slurm

# Monitor
squeue -u $USER
tail -f logs/omixai_<JOBID>.out

# Step 2 — permutation feature importance (Reviewer 1 comparison).
# Same runner, PFI format. Parallelises across GPUs in pure Python.
METHOD=pfi sbatch scripts/omixai.slurm
#   or directly:
#   python scripts/run_interpret.py --model "$WEIGHTS" --data_dir "$DATA_DIR" \
#       --method pfi --out_dir results/pfi_dl/

# Step 3 — retrain with top-k features. Run after the matching interpretation.
# --k takes one value or a comma list ("all" = full feature set).
python scripts/run_retrain.py --data_dir "$DATA_DIR" \
    --ranking results/omixai_ranking.csv \
    --k 50,100,300,500,700,all                    # OmiXAI arm (sequential)
python scripts/run_retrain.py --data_dir "$DATA_DIR" \
    --pfi_scores results/pfi_dl/pfi_dl_scores.npy \
    --feature_names results/feature_names.json --k 50,100,300,500,700,all   # PFI arm

# Parallel across k (one process per GPU) — needs a multi-GPU allocation:
#   #SBATCH --gres=gpu:v100:2
# python scripts/run_retrain.py --data_dir "$DATA_DIR" \
#     --ranking results/omixai_ranking.csv --k 50,100,300,500,700,all --parallel
```

Results are written to `results/`:

| File | Contents |
|------|----------|
| `omixai_ranking.csv` | Hybrid-ranked feature list |
| `omixai_gnn_scores.npy` | Raw attribution scores per method |
| `feature_names.json` | Canonical feature order (shared by all scripts) |
| `pfi_dl/pfi_dl_scores.npy` | DL permutation importance scores |
| `retrain_omixai.csv` / `retrain_pfi.csv` | F1 / AUC at top-k per arm |

---

## Quick start (Python API)

```python
from omixai import OmiXAI

# Pass an already-trained model. model_type is auto-detected from layer types
# (Conv2d → cnn, MessagePassing → gnn). No feature counts are needed: the input
# width is read from the data, and the number of leading channels to skip is
# inferred as F - len(feature_names) at ranking time.
pipeline = OmiXAI(model=graph_model)

# Hybrid ensemble (default). Interpret train TPs only — test set stays sealed.
pipeline.interpret(train_loader, method="hybrid", width=100)
rankings = pipeline.rank_features(feature_names=feature_list)   # DNA channels dropped here
print(rankings.head(20))

# Permutation feature importance (GNN), parallel across GPUs in pure Python.
pipeline.interpret(train_loader, method="pfi", pfi_batch_size=64)
pfi_rank = pipeline.rank_features(feature_names=feature_list)
```

Non-genomic use: just pass `feature_names` covering every channel — then nothing
is skipped (`skip = F - len(feature_names) = 0`).

---

## Repository structure

```
OmiXAI/
├── omixai/
│   ├── __init__.py
│   ├── pipeline.py          # OmiXAI class — interpret(method=...) + rank_features()
│   ├── models/
│   │   ├── cnn.py           # ConvMZC
│   │   └── gnn.py           # GraphMZC
│   ├── data/
│   │   ├── dataset.py       # GenomicDataset + split (CNN)
│   │   ├── graph_dataset.py # GraphGenomicDataset + stratified_split_intervals (GNN)
│   │   └── genome.py        # genome loading + one-file joblib cache
│   ├── xai/
│   │   └── pfi.py           # batched, multi-GPU permutation feature importance
│   └── training/
│       ├── train_cnn.py     # training loop + metrics
│       ├── train_gnn.py
│       └── retrain.py       # retrain_topk: top-k feature-reduction experiment
├── scripts/                 # thin runners (no logic) + SLURM wrappers
│   ├── run_interpret.py     # interpretation runner (--method hybrid|pfi)
│   ├── run_retrain.py       # top-k retraining runner
│   ├── compare_old_new_ranking.py
│   ├── genome_cache.py      # back-compat shim → omixai.data.genome
│   └── omixai.slurm         # SLURM job (METHOD=hybrid|pfi)
├── notebooks/
├── results/
├── README.md
├── pyproject.toml          # package metadata + dependencies (pip install -e .)
└── requirements.txt
```

---

## Citation

```bibtex
@article{Alaeva2025omixai,
  author       = {Alaeva, Ameliia and Lapteva, Anna and Mikhaylovskaya, Natalya
                  and Malkov, Vladislav and Herbert, Alan
                  and Borevskiy, Andrey and Poptsova, Maria},
  title        = {OmiXAI: An Ensemble XAI Pipeline for Interpretable
                  Deep Learning in Omics Data},
  elocation-id = {2025.04.28.651097},
  year         = {2025},
  doi          = {10.1101/2025.04.28.651097},
  publisher    = {Cold Spring Harbor Laboratory},
  url          = {https://www.biorxiv.org/content/10.1101/2025.04.28.651097v1},
  eprint       = {https://www.biorxiv.org/content/10.1101/2025.04.28.651097v1.full.pdf},
  journal      = {bioRxiv}
}
```
