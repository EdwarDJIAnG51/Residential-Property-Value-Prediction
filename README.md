# Residential Property Value Prediction — Kaggle Competition

**Washington University in St. Louis | T81-558: Applications of Deep Neural Networks**

**Team Prompt** · Chufan Jiang · Xinyue Ha · Linyuan Zhao · Yutian Han

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue?logo=kaggle)](https://www.kaggle.com/competitions/app-of-gen-ai-deep-learning-wustl-fall-2025)

---

## Overview

Predict a **property value score (0–100)** for residential homes using a rich multi-modal dataset that combines structured tabular features, free-text property descriptions, and exterior photographs.

**Evaluation Metric:** Root Mean Square Error (RMSE) — lower is better.

---

## Our Approach

We built a **multi-modal stacking ensemble** that extracts signals from all three data modalities and combines them through a Ridge meta-learner.

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Input Data                             │
│  ┌─────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │  Tabular    │  │  Text (desc.)    │  │   Images    │ │
│  │  Features   │  │  TF-IDF 30k      │  │  CLIP ViT   │ │
│  └──────┬──────┘  └────────┬─────────┘  └──────┬──────┘ │
└─────────┼──────────────────┼────────────────────┼────────┘
          │                  │                    │
    ┌─────▼──────┐    ┌──────▼──────┐    ┌────────▼───────┐
    │  XGBoost   │    │  LightGBM   │    │  MultiModal    │
    │  LightGBM  │    │  (Text)     │    │  MLP (Tab +    │
    │  CatBoost  │    │             │    │  CLIP Image)   │
    └─────┬──────┘    └──────┬──────┘    └────────┬───────┘
          │                  │                    │
          └──────────────────┼────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Ridge Meta-    │
                    │  Learner Stack  │
                    └────────┬────────┘
                             │
                    Final Prediction
```

### Modality Handling

| Modality | Technique | Details |
|---|---|---|
| Structured (tabular) | Feature engineering + tree models | XGBoost, LightGBM, CatBoost with 5-fold OOF |
| Text descriptions | TF-IDF + LightGBM | 30,000 TF-IDF features on property descriptions |
| Images (exterior photos) | CLIP ViT-B/32 | 512-dim embeddings extracted from property photos |

### Base Models (5-Fold Cross-Validation)

| Model | OOF RMSE |
|---|---|
| CatBoost (tabular) | 3.3585 |
| LightGBM (tabular) | 3.4657 |
| XGBoost (tabular) | — |
| MultiModal MLP (tabular + CLIP images) | 3.6800 |
| LightGBM (TF-IDF text) | 4.3358 |

### Stacking (Meta-Learner)

Out-of-fold predictions from all 5 base models were stacked and passed to a **Ridge regression** meta-learner:

- **Final Validation RMSE: 3.2588**
- Meta weights learned: `XGB(-2.55) · LGB(1.23) · CatBoost(5.64) · MultiModal(1.11) · Text(1.68)`

### MultiModal MLP Architecture

```
Tabular branch:   Linear(n→256) → ReLU → Dropout(0.2) → Linear(256→128) → ReLU
Image branch:     Linear(512→256) → ReLU → Dropout(0.2)
Fusion:           Linear(128+256→128) → ReLU → Dropout(0.2) → Linear(128→1)
```

---

## Tech Stack

- **ML Frameworks:** PyTorch, scikit-learn, XGBoost, LightGBM, CatBoost
- **Vision:** OpenAI CLIP (ViT-B/32)
- **NLP:** scikit-learn TF-IDF Vectorizer
- **Data:** pandas, NumPy
- **Training:** Google Colab (GPU)

---

## Key Techniques

- **5-fold cross-validation** with out-of-fold (OOF) predictions to prevent leakage into the meta-learner
- **CLIP image embeddings** — zero-shot visual feature extraction from exterior property photos
- **TF-IDF (30k features)** — captures vocabulary from free-text descriptions of interiors, exteriors, and architecture
- **Early stopping** (patience=8) on the MultiModal MLP to prevent overfitting
- **StandardScaler** applied to meta-features before Ridge stacking

---

## Repository Structure

```
├── Team Prompt Kaggle code.ipynb   # Full pipeline: EDA → feature engineering → modeling → submission
└── README.md
```

---

## Competition

- **Course:** T81-558 Applications of Deep Neural Networks / T81-559 Applications of Generative AI — Washington University in St. Louis (Fall 2025)
- **Instructor:** Jeff Heaton
- **Kaggle:** [app-of-gen-ai-deep-learning-wustl-fall-2025](https://www.kaggle.com/competitions/app-of-gen-ai-deep-learning-wustl-fall-2025)
