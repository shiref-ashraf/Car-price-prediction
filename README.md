# Used Car Price Prediction using Deep Learning

## Project Overview
This repository contains a complete deep learning pipeline for used car price prediction.

## Repository Structure

```text
project/
│
├── data/
├── notebooks/
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── models/
├── reports/
├── presentation/
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python src/train.py
```

## Evaluation

```bash
python src/evaluate.py
```

## Reproducibility
- Fixed random seed
- Leakage-free preprocessing
- Standardized training pipeline
