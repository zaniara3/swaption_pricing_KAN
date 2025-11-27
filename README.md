# Swaption Pricing with FastKAN

This repository contains code for **approximating swaption prices** using **Kolmogorov–Arnold Networks (KANs)**—specifically the **FastKAN** implementation—trained on Monte Carlo–simulated swaption datasets.

The goal is to build a **fast, accurate surrogate model** for swaption pricing using yield-curve factors and time-to-maturity as inputs.

## Project Structure

```
.
├── data/
│   └── samples_to_trainNN_al60_be180_k25083_forgrad.pkl   # Training dataset
│
├── models/
│   └── swaption_network_al60_be180_k25083_forgrad.pt   # Trained model artifacts
│
├── src/
│   ├── train_swpation_prices_NN__al60_be180_k25083_fastkan.py   # FastKAN training script
│   └── generate_random_samples_for_network.py                # Monte Carlo simulation / dataset generator
│
└── README.md
```

## Overview

Swaptions (European-style options on interest-rate swaps) are priced here using a two-stage pipeline:

### 1. Data Generation
`generate_random_samples_for_network.py`  
- Simulates yield-curve factors under a DTAFNS multi-factor model  
- Generates swaption payoffs via Monte Carlo simulation  
- Saves results to a `.pkl` file for training  

### 2. Neural Network Pricing Model
`train_swpation_prices_NN__al60_be180_k25083_fastkan.py`  
- Loads the dataset  
- Builds a **FastKAN** model  
- Trains the surrogate pricing network  
- Uses GPU automatically when available  
- Saves trained model artifacts  

## Why FastKAN?
 
- More sample-efficient  
- Much faster to train  
- Better suited to smooth but nonlinear pricing functions  

## Installation

```bash
git clone https://github.com/zaniara3/swaption_pricing_KAN.git
cd swaption_pricing_KAN
```
## Create and activate a virtual environment (recommended)

#### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```
#### Windows (PowerShell)
```bash
python -m venv .venv
.venv\Scripts\activate
```

## Usage
### Install packages
```bash
pip install -r requirements.txt
```
### Generate Training Samples
```bash
python src/generate_random_samples_for_network.py
```

### Train the FastKAN Model
```bash
python src/train_swpation_prices_NN__al60_be180_k25083_fastkan.py
```

## Research Context

This code supports research in:
- Machine-learning derivative pricing  
- Surrogate modeling for hedging  
- Neural interest rate models  
- FastKAN architectures  


