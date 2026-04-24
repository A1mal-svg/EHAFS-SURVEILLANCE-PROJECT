# EHAFS — Efficient Hybrid-Attention Framework for Surveillance

Real-time video action recognition for CCTV surveillance, combining **MobileNetV3-Small** + **Temporal Shift Module (TSM)** + **Frame-level Temporal Attention**, trained and evaluated on the **RWF-2000** dataset (Fight vs NonFight).

> Authors: Aimal Khan & Muhammad Zain
> Paper: *Efficient Hybrid-Attention Framework for Surveillance (EHAFS)*

---

## ✨ Features

- 🧠 **Hybrid CNN–Attention** model: MobileNetV3-Small backbone + TSM + temporal-attention head (~3.4M params).
- 🎯 Trained on **RWF-2000** (Fight vs NonFight) — no dummy data, real benchmark.
- 🖥️ **Streamlit web app**: upload a CCTV clip → see prediction, confidence, and **per-frame attention heatmap**.
- 🧪 Reproducible: deterministic seed, AdamW, cosine LR, 16-frame uniform sampling, 224×224, ImageNet normalization.
- 🛠️ Runs locally in **VS Code / any editor**, deployable to **Streamlit Community Cloud**, **Hugging Face Spaces**, **Render**, or any Docker host.

---

## 📁 Project structure

```
ehafs/
├── app/
│   └── streamlit_app.py        # Web UI (upload video → predict + attention)
├── ehafs/
│   ├── __init__.py
│   ├── model.py                # MobileNetV3 + TSM + Temporal Attention
│   ├── tsm.py                  # Temporal Shift Module
│   ├── attention.py            # Frame-level attention head
│   ├── dataset.py              # RWF-2000 video dataset
│   ├── transforms.py           # Augmentation + normalization
│   ├── inference.py            # Load checkpoint, predict, return attention
│   └── utils.py                # Seeding, metrics, config loader
├── scripts/
│   ├── train.py                # Training entry point
│   ├── evaluate.py             # Test-set evaluation (Acc/Prec/Rec/F1)
│   ├── download_rwf2000.py     # Helper to fetch RWF-2000
│   └── prepare_rwf2000.py      # Verify/normalize folder layout
├── configs/
│   └── default.yaml            # All hyperparameters
├── data/                       # Place RWF-2000 here (gitignored)
├── notebooks/
│   └── visualize_attention.ipynb
├── requirements.txt
├── Dockerfile
├── .streamlit/config.toml
└── README.md
```

---

## 🚀 Quick start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the RWF-2000 dataset

The dataset is **not redistributable** through this repo. Use the helper:

```bash
python scripts/download_rwf2000.py
```

This walks you through the official sources (Kaggle / authors' Google Drive) and verifies the expected layout:

```
data/RWF-2000/
├── train/
│   ├── Fight/      (800 .avi/.mp4)
│   └── NonFight/   (800 .avi/.mp4)
└── val/
    ├── Fight/      (200 .avi/.mp4)
    └── NonFight/   (200 .avi/.mp4)
```

If you already extracted it, just point the config to it:

```bash
python scripts/prepare_rwf2000.py --data_dir /path/to/RWF-2000
```

### 3. Train

```bash
python scripts/train.py --config configs/default.yaml
```

Outputs checkpoints to `checkpoints/ehafs_best.pt` and TensorBoard logs to `runs/`.

### 4. Evaluate

```bash
python scripts/evaluate.py --checkpoint checkpoints/ehafs_best.pt
```

Reports Accuracy, Precision, Recall, F1, and confusion matrix on the RWF-2000 val split.

### 5. Launch the web app

```bash
streamlit run app/streamlit_app.py
```

Open <http://localhost:8501>, drag in a CCTV clip (`.mp4` / `.avi` / `.mov`), and inspect the prediction + attention heatmap.

---

## 🧱 Model architecture

```
Input clip  [B, T=16, 3, 224, 224]
        │
        ▼
MobileNetV3-Small (ImageNet pretrained)
   └─ TSM inserted after each inverted-residual block (1/8 of channels shifted ±1 along T)
        │
        ▼
Per-frame feature  f_t ∈ R^{576}
        │
        ▼
Temporal Attention:
   s_t = w·f_t          (linear, no bias)
   α_t = softmax_t(s_t)
   v   = Σ_t α_t · f_t
        │
        ▼
Linear classifier → 2 logits (Fight / NonFight)
```

**Why this matters**: TSM gives 2D conv pseudo-3D temporal modeling at zero FLOP cost, and the attention layer focuses on the few frames where action actually happens — both essential for low-latency surveillance.

---

## 📊 Reported results (RWF-2000 val)

| Model              | Params (M) | RWF Acc | RWF F1 | Latency (Jetson NX) |
| ------------------ | ---------- | ------- | ------ | ------------------- |
| ResNet3D-50        | 33.0       | 85.0%   | 0.85   | 65 ms               |
| MobileNetV3 + TSM  | 8.5        | 87.8%   | 0.88   | 20 ms               |
| **EHAFS (ours)**   | **3.4**    | **90.2%** | **0.90** | **30 ms**       |

---

## 🐳 Deploy

### Streamlit Community Cloud
Push to GitHub → connect on <https://share.streamlit.io> → entry point `app/streamlit_app.py`.

### Hugging Face Spaces (Streamlit SDK)
`README` metadata + `app/streamlit_app.py`.

### Docker / Render / Railway
```bash
docker build -t ehafs .
docker run -p 8501:8501 ehafs
```

### Local IDEs
Open the folder in **VS Code** or **any editor**, use the Python interpreter from `.venv`, run `streamlit run app/streamlit_app.py` from the integrated terminal.

> **Vercel** is JS/serverless-only — Streamlit/PyTorch won't run there. Use Streamlit Cloud, HF Spaces, Render, Railway, or Fly.io for the Python app.

---

## 📜 Citation

```bibtex
@misc{khan2025ehafs,
  title  = {Efficient Hybrid-Attention Framework for Surveillance (EHAFS)},
  author = {Khan, Aimal and Zain, Muhammad},
  year   = {2025}
}
```

References: RWF-2000 [Cheng et al., 2020], TSM [Lin et al., 2019], Temporal Attention [Meng et al., 2019], MobileNetV3 [Howard et al., 2019].
