# Used Car Price Prediction using Deep Learning

## Project Overview
This repository contains a complete deep learning pipeline for used car price prediction, ranging from data preprocessing and model training to evaluation and a live web application demo.


## Installation

```bash
pip install -r requirements.txt
```

## Live Web Application Demo

You can run the model locally using the provided Flask web application. It serves a beautiful frontend UI (`CarPrice_Live_Demo.html`) and connects it to the locally trained Keras model via a `/predict` API endpoint.

1. Start the Flask server:
```bash
python app.py
```
2. Open your web browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000).
3. Configure your vehicle parameters and click **Predict Price** to see the Deep Learning model's real-time estimation!

## Training

To train the model from scratch on the dataset:
```bash
python srs/train.py
```

## Evaluation

To evaluate the trained model on the hold-out test set:
```bash
python srs/evaluate.py
```

## Inference (CLI)

You can also run predictions from the command line on new data:
```bash
python srs/inference.py
# Or on a batch CSV:
python srs/inference.py --csv path/to/new_cars.csv
```

## Reproducibility
- Fixed random seed
- Leakage-free preprocessing
- Standardized training pipeline
