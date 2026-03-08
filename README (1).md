# 📱 Hybrid Model for Real-Time Mobile Snatching Detection in Video Surveillance

<div align="center">

![Python](https://img.shields.io/badge/Python-3.6-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-v0.1.11-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-v3.0.2-red?style=flat-square&logo=keras)
![License](https://img.shields.io/badge/License-CC%20BY%203.0-green?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-96.45%25-brightgreen?style=flat-square)
![Journal](https://img.shields.io/badge/Published-VFAST%20TSE%202026-purple?style=flat-square)

**Time-Distributed CNN + Attention-Based LSTM for Mobile Theft Detection**

[📄 Paper](#citation) • [📦 Dataset](#dataset) • [🚀 Quick Start](#quick-start) • [📊 Results](#results) • [🤝 Contributing](#contributing)

</div>

---

## 📌 Overview

This repository contains the official implementation of the paper:

> **"Hybrid Model for Real-Time Mobile Snatching Detection in Video Surveillance Using Time-Distributed CNN and Attention-Based LSTM"**
> Faisal Khan, Irshad Ahmad, Muhammad Zubair*, Yasir Saleem Afridi
> *VFAST Transactions on Software Engineering*, Vol. 14, No. 1, pp. 19–35, February 2026
> 🔗 **DOI:** [10.21015/vtse.v14i1.2279](https://doi.org/10.21015/vtse.v14i1.2279)
> 🌐 **Article:** [vfast.org — Article #2279](https://vfast.org/journals/index.php/VTSE/article/view/2279)
> 🆔 **Article ID:** 2279 | **ISSN(e):** 2309-3978 | **ISSN(p):** 2411-6246

Mobile phone snatching is a rapidly growing street crime in Pakistan and globally. This project proposes a **hybrid deep learning model** (TD_CNN-LSTM) that combines:
- 🧠 **Time-Distributed CNNs** for per-frame spatial feature extraction
- 🔁 **Attention-Based LSTM** for temporal dependency modeling across video sequences
- 🎯 **Attention Mechanism** to focus on the most salient cues in video frames

The model achieves **96.45% accuracy** on a real-world mobile snatching dataset collected from social media platforms.

---

## 🏗️ Model Architecture

```
Input Video Frames
       │
       ▼
┌─────────────────────────┐
│  Time-Distributed CNN   │  ← Spatial feature extraction per frame
│  (3 Conv Blocks)        │    Filters: 16 → 32 → 64
│  + BatchNorm + MaxPool  │    Input tensor: 10 × 240 × 240 × 3
└────────────┬────────────┘
             │ Feature maps: 30 × 30 × 32
             ▼
┌─────────────────────────┐
│     LSTM Layer          │  ← Temporal sequence modeling
│     (10 units)          │    Long-range dependency capture
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Attention Mechanism   │  ← Weighted focus on salient frames
│   (Softmax weights)     │    Context vector computation
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Dense (128) + ReLU    │
│   Dropout (0.7)         │
│   Dense (1) + Sigmoid   │
└────────────┬────────────┘
             │
             ▼
    Snatching / Normal
```

---

## 📂 Repository Structure

```
mobile-snatching-detection/
│
├── 📄 README.md                        # This file
├── 📄 CITATION.md                      # All citation formats (BibTeX, APA, IEEE, MLA, Chicago)
├── 📄 requirements.txt                 # Python dependencies
├── 📄 LICENSE                          # CC BY 3.0
│
├── 📁 data/
│   ├── README.md                       # Dataset description & download link
│   ├── sample_snatching/               # Sample snatching video clips
│   └── sample_normal/                  # Sample normal video clips
│
├── 📁 src/
│   ├── preprocess.py                   # Frame extraction, normalization, cropping
│   ├── model.py                        # TD_CNN-LSTM model definition
│   ├── attention.py                    # Custom attention layer
│   ├── train.py                        # Training script
│   ├── evaluate.py                     # Evaluation & metrics
│   └── predict.py                      # Inference on new videos
│
├── 📁 notebooks/
│   ├── 01_data_exploration.ipynb       # Dataset analysis & visualization
│   ├── 02_model_training.ipynb         # Training walkthrough
│   ├── 03_transfer_learning.ipynb      # VGG19 / ResNet50 / InceptionV3 comparison
│   └── 04_results_visualization.ipynb  # Confusion matrix, loss curves
│
├── 📁 models/
│   └── README.md                       # Instructions to download pretrained weights
│
├── 📁 results/
│   ├── confusion_matrix.png            # Test set confusion matrix
│   ├── training_curves.png             # Accuracy & loss over 50 epochs
│   └── comparison_table.png            # Comparison with baseline models
│
└── 📁 paper/
    └── VTSE_2279.pdf                   # Published paper (open access)
```

---

## 🔬 Method Details

### Data Preprocessing

| Step | Description |
|------|-------------|
| **Frame Extraction** | Videos standardized to 64 frames; padding for short clips, stride-3 sampling for long clips (>160 frames) |
| **Normalization** | Min-max pixel normalization: `I_norm(x,y) = (I(x,y) - I_min) / (I_max - I_min)` |
| **Cropping & Resize** | Each frame resized to **256×256**, then cropped to **240×240** pixels |
| **Sequence Length** | 10 frames per sequence; 3 RGB channels |

### Model Components

**Time-Distributed CNN**
- 3 convolutional blocks with increasing filter counts (16, 32, 64)
- ReLU activation + Batch Normalization after each block
- Max-Pooling for spatial downsampling
- Dropout for regularization
- Output feature maps: `30 × 30 × 32` → flattened to `28,800 × 1`

**Attention-Based LSTM**
- LSTM with input, forget, cell, and output gates (Hochreiter & Schmidhuber, 1997)
- Attention weights computed via feed-forward network with `tanh` activation
- Softmax normalization of attention scores
- Context vector as weighted sum of hidden states

**Classification Head**
- Dense layer: 128 units + ReLU
- Dropout rate: 0.7
- Output: 1 unit + Sigmoid (binary: Snatching / Normal)

---

## 📦 Dataset

The **Mobile Snatching Dataset** consists of **200 videos** (100 snatching + 100 normal) collected from:
- YouTube, TikTok, Twitter (X), Facebook, Google
- Publicly available surveillance footage

| Split | Videos | Snatching | Normal |
|-------|--------|-----------|--------|
| Train | 140 (70%) | 70 | 70 |
| Validation | 40 (20%) | 20 | 20 |
| Test | 20 (10%) | 10 | 10 |

📥 **Download Dataset & Source Code:**
[Google Drive Link](https://drive.google.com/drive/folders/130rbkDPgf-ixJFfOGgDxIdhRraT3fKfZ?usp=sharing)

> Videos are resized to **240×240 pixels** with a sequence length of **10 frames**.

---

## 🚀 Quick Start

### Prerequisites

```bash
git clone https://github.com/YOUR_USERNAME/mobile-snatching-detection.git
cd mobile-snatching-detection
pip install -r requirements.txt
```

### Requirements

```
tensorflow==0.1.11
keras==3.0.2
numpy>=1.19.0
opencv-python>=4.5.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

### 1. Preprocess Videos

```python
from src.preprocess import preprocess_dataset

preprocess_dataset(
    input_dir='data/raw_videos/',
    output_dir='data/processed/',
    num_frames=64,
    img_size=(240, 240)
)
```

### 2. Train the Model

```python
from src.train import train_model

train_model(
    data_dir='data/processed/',
    epochs=50,
    batch_size=16,
    learning_rate=0.0001,
    save_path='models/td_cnn_lstm.h5'
)
```

### 3. Evaluate

```python
from src.evaluate import evaluate_model

results = evaluate_model(
    model_path='models/td_cnn_lstm.h5',
    test_dir='data/processed/test/'
)
# Outputs: Accuracy, Precision, Recall, Confusion Matrix
```

### 4. Predict on a New Video

```python
from src.predict import predict_video

result = predict_video(
    model_path='models/td_cnn_lstm.h5',
    video_path='your_video.mp4'
)
print(f"Prediction: {result}")  # 'Snatching' or 'Normal'
```

---

## 📊 Results

### Performance vs. Transfer Learning Baselines

| Model | Accuracy |
|-------|----------|
| VGG19 + LSTM | 87.27% |
| ResNet50 + LSTM | 91.29% |
| InceptionV3 + LSTM | 94.45% |
| **TD-CNN-LSTM (Ours)** | **96.45%** |

### Comparison with State-of-the-Art

| Model | Config | Epochs | Avg. Time (s) | Accuracy |
|-------|--------|--------|---------------|----------|
| LanHAR [2024] | RGB | 100 | 0.41 | 72.10% |
| Full Transformer Network [2022] | RGB | 100 | 0.0521 | 85.72% |
| SPOTER [2022] | Pose | 100 | 0.034 | 73.53% |
| MIPA-ResGCN [2023] | Pose | 150 | 0.0491 | 85.43% |
| SIGNGRAPH [2023] | Pose | 150 | 0.0521 | 84.01% |
| Hybrid CNN-LSTM [2024] | RGB | 100 | 0.0307 | 88.32% |
| **TD-CNN-LSTM + Attention (Ours)** | **RGB** | **50** | **0.0371** | **96.45%** |

### Confusion Matrix (Test Set)

|  | Predicted: Snatching | Predicted: Normal |
|--|----------------------|-------------------|
| **Actual: Snatching** | 115 ✅ | 5 ❌ |
| **Actual: Normal** | 2 ❌ | 118 ✅ |

---

## 🧪 Experimental Setup

| Component | Specification |
|-----------|---------------|
| CPU | Intel Core i7-9700F |
| GPU | NVIDIA RTX 2080 Super |
| RAM | 16 GB |
| Framework | TensorFlow (backend) + Keras |
| Python | 3.6 |
| Optimizer | Adam (lr=0.0001) |
| Batch Size | 16 |
| Epochs | 50 |

---

## 🗺️ Snatching Types Covered

| Snatching Type | Victim Reaction |
|----------------|-----------------|
| Distraction and grab | Fights back |
| Grab and run | Yells for help |
| Fake accident and grab | Calls the police |
| Group snatching | Tries to fight back |
| Snatching from a vehicle | Sounds alarm |

---

## 📈 Mobile Snatching Statistics in Pakistan

| Year | City | Phones Snatched |
|------|------|-----------------|
| 2022 | Karachi | 19,000+ |
| 2023 | Karachi | 29,536 (Jan–Oct) |
| 2019–23 | Lahore | 14,000 |
| 2023 | Islamabad | 680+ |
| 2023 | Peshawar | 1,750+ |

---

## ⚠️ Limitations

- Model not fully tested across diverse urban environments
- May underperform in high-occlusion or low-quality footage scenarios
- Real-time deployment requires significant GPU resources

## 🔮 Future Work

- Scalable methodologies for varied urban settings
- Edge computing integration for processing efficiency
- Extended testing on additional crime categories
- Advanced neural architectures (e.g., Vision Transformers)
- Larger and more diverse dataset collection

---

## 📝 Citation

If you use this code, dataset, or findings from this paper in your research, please cite:

### BibTeX (Official)

```bibtex
@article{Khan_Ahmad_Zubair_Afridi_2026,
  title        = {Hybrid Model for Real-Time Mobile Snatching Detection in Video Surveillance
                  Using Time-Distributed CNN and Attention-Based LSTM},
  author       = {Khan, Faisal and Ahmad, Irshad and Zubair, Muhammad and Afridi, Yasir Saleem},
  journal      = {VFAST Transactions on Software Engineering},
  volume       = {14},
  number       = {1},
  pages        = {19--35},
  year         = {2026},
  month        = {Feb.},
  doi          = {10.21015/vtse.v14i1.2279},
  url          = {https://vfast.org/journals/index.php/VTSE/article/view/2279}
}
```

### APA Format

Khan, F., Ahmad, I., Zubair, M., & Afridi, Y. S. (2026). Hybrid Model for Real-Time Mobile Snatching Detection in Video Surveillance Using Time-Distributed CNN and Attention-Based LSTM. *VFAST Transactions on Software Engineering*, *14*(1), 19–35. https://doi.org/10.21015/vtse.v14i1.2279

### IEEE Format

F. Khan, I. Ahmad, M. Zubair, and Y. S. Afridi, "Hybrid Model for Real-Time Mobile Snatching Detection in Video Surveillance Using Time-Distributed CNN and Attention-Based LSTM," *VFAST Transactions on Software Engineering*, vol. 14, no. 1, pp. 19–35, Feb. 2026. DOI: 10.21015/vtse.v14i1.2279

### Paper Details

| Field | Info |
|-------|------|
| **Article ID** | 2279 |
| **DOI** | [10.21015/vtse.v14i1.2279](https://doi.org/10.21015/vtse.v14i1.2279) |
| **Journal URL** | [vfast.org/journals/index.php/VTSE/article/view/2279](https://vfast.org/journals/index.php/VTSE/article/view/2279) |
| **Journal** | VFAST Transactions on Software Engineering |
| **Volume / Issue** | 14 / 1 |
| **Pages** | 19–35 |
| **Published** | February 2026 |
| **Submitted** | November 06, 2025 |
| **Accepted** | February 08, 2026 |
| **ISSN (Online)** | 2309-3978 |
| **ISSN (Print)** | 2411-6246 |
| **License** | CC BY 3.0 |

> 📄 See [CITATION.md](./CITATION.md) for all citation formats and full abstract.

---

## 👥 Authors

| Author | Affiliation |
|--------|-------------|
| **Faisal Khan** | Dept. of Computer Science, Islamia College Peshawar, KPK, Pakistan |
| **Irshad Ahmad** | Dept. of Computer Science, Islamia College Peshawar, KPK, Pakistan |
| **Muhammad Zubair** ✉️ | Dept. of Computer Science, Islamia College Peshawar, KPK, Pakistan |
| **Yasir Saleem Afridi** | Dept. of Computer Systems Engineering, UET Peshawar, Pakistan |

📧 Correspondence: [zubair@icp.edu.pk](mailto:zubair@icp.edu.pk)

---

## 📄 License

This work is licensed under a [Creative Commons Attribution 3.0 License](https://creativecommons.org/licenses/by/3.0/).

---

## 🙏 Acknowledgments

This research received no external funding. The authors thank the open-source community and the researchers whose datasets and frameworks made this work possible.

---

<div align="center">
⭐ If you find this work useful, please consider starring the repository!
</div>
