# AI-PK-Predictor

AI-PK-Predictor is a deep learning-based tool designed to estimate individual pharmacokinetic (PK) parameters — including half-life, clearance, and AUC — using only minimal clinical features such as age, weight, sex, liver function, and creatinine levels.

## Features

- Lightweight neural network implemented in TensorFlow
- Trained on synthetic data generated using pharmacokinetic equations
- Predicts personalized PK values based on 5 input features
- Suitable for low-resource settings or early-phase simulations

## Input Features

- Age (in years)
- Sex (0 = Female, 1 = Male)
- Weight (in kilograms)
- Liver score (numeric proxy)
- Creatinine (in mg/dL)

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
