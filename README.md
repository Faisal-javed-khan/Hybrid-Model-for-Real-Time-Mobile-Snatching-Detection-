# Dataset — Mobile Snatching

## Overview

The **Mobile Snatching Dataset** is a custom dataset collected specifically for this research. It contains **200 videos** of real-world mobile phone snatching incidents and normal activities.

## Download

📥 **Dataset & Source Code:** [Google Drive](https://drive.google.com/drive/folders/130rbkDPgf-ixJFfOGgDxIdhRraT3fKfZ?usp=sharing)

## Composition

| Class | Videos | Source |
|-------|--------|--------|
| Snatching | 100 | YouTube, TikTok, Twitter, Facebook, Google |
| Normal | 100 | Public surveillance, social media |
| **Total** | **200** | — |

## Snatching Types

| Type | Description |
|------|-------------|
| Distraction and grab | Offender distracts victim and grabs phone |
| Grab and run | Quick snatch followed by escape on foot |
| Fake accident and grab | Staged collision used as cover |
| Group snatching | Multiple offenders overwhelming victim |
| Snatching from vehicle | Theft from a moving motorcycle or car |

## Preprocessing Applied

- All videos resized to **240 × 240 pixels**
- Standardized to **10-frame sequences**
- Min-max normalized per frame
- Split: 70% train / 20% val / 10% test

## Directory Structure (after download)

```
data/
├── raw_videos/
│   ├── snatching/   ← 100 snatching videos (.mp4)
│   └── normal/      ← 100 normal videos (.mp4)
└── processed/
    ├── snatching/   ← preprocessed .npy files
    └── normal/
```
