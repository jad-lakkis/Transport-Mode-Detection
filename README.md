# Federated Multi-City Transport Mode Detection

In federated multi-city transport mode detection, each city has a different local data distribution and often different class imbalance. 
Because of this, optimizing standard cross-entropy may not align well with the metric of interest, especially for minority transport modes.
Since AUPRC/AP is more appropriate for imbalanced classification, we propose to replace the usual cross-entropy objective with a PR-oriented surrogate. 
However, the SOAP method is formulated for binary classification, so we adapt it to the multiclass setting through one-vs-rest binarization: for each transport class, we treat that class as positive and all other classes as negative, optimize a SOAP-like loss locally on each city, and aggregate the model updates in a federated learning framework. We then evaluate the global model using per-class AUPRC and macro-AUPRC, rather than relying only on accuracy.
We will first see if this approach outperforms cross-entropy and then apply Federated Learning (the local models would be on Mekkah, KAUST,Jeddah, and Kazakhstan).


# CellMob Transport Mode Classification

This repository contains Python scripts used to train and evaluate RNN-based models for transport mode classification using cellular signal measurements.

Target classes:

- **walk**
- **bus**
- **car**

The scripts train models on sequential windows of cellular signal measurements and evaluate them using metrics such as:

- Accuracy
- Confusion Matrix
- Precision–Recall curves
- AP / AUPRC per class
- Macro AUPRC

---

# 1. System Requirements

This project was developed using:

- **Python 3.9+**
- **Windows**

All commands below assume a **Windows environment** using **PowerShell** or **Command Prompt**.

---

# 2. Create the Project Folder
## Data Setup

After cloning the repository, you must place the **downloaded raw dataset** inside the project structure exactly as described below.

 **note:**  
All **raw data that you download must be placed inside the folder**

Each subfolder inside `orignal_raw_data/` corresponds to a **specific city and transportation mode**.  
These folders contain the **original logs exactly as downloaded**, before any preprocessing.

---

## Expected Project Structure
## Expected Project Structure

After cloning or copying the repository, your folder should look like this:

```text
fed-auprc-tmd/
└── CellMob/
    ├── zcodes/
    │
    └── Data/
        │
        ├── 6400 KAUST/                         # preprocessed KAUST dataset used by the experiments (codes available)
        │   ├── bus_test_kaust_standardized_6400windows.csv
        │   ├── bus_train_kaust_standardized.csv
        │   ├── car_test_kaust_standardized_6400windows.csv
        │   ├── car_train_kaust_standardized.csv
        │   ├── walk_test_kaust_standardized_6400windows.csv
        │   └── walk_train_kaust_standardized.csv
        │
        ├── orignal_raw_data/                   # ORIGINAL data you must download
        │   ├── bus_colored_kaust/
        │   ├── bus_jeddah/
        │   ├── bus_mekkah/
        │   ├── car_jeddah/
        │   ├── car_kaust/
        │   ├── car_kz/
        │   ├── car_mekkah/
        │   ├── train_mekkah/
        │   ├── walk_jeddah/
        │   ├── walk_kaust/
        │   ├── walk_kz/
        │   └── walk_mekkah/
        │
        ├── zdata_unfinished/                   # intermediate cleaned CSV files generated from raw logs
        │   ├── bus_colored_kaust_cleaned.csv
        │   ├── bus_jeddah_cleaned.csv
        │   ├── bus_mekkah_cleaned.csv
        │   ├── car_jeddah_cleaned.csv
        │   ├── car_kaust_cleaned.csv
        │   └── ...
        │
        └── data(raw_but_seperated)/
            ├── zdata_train/                    # standardized train files (80%)
            │   ├── walk_kaust_cleaned.csv
            │   ├── bus_colored_kaust_cleaned.csv
            │   └── car_kaust_cleaned.csv
            │
            └── zdata_test/                     # standardized test files (20%)
                ├── walk_kaust_cleaned.csv
                ├── bus_colored_kaust_cleaned.csv
                └── car_kaust_cleaned.csv
```

The scripts must remain **inside the `CellMob` folder**, because the paths in the code are relative.

---

# 3. Data Directory Structure


Two dataset formats are used.

---

## 3.1 Imbalanced Dataset Format

Used for most experiments.

---

## 3.2 Fixed Balanced Dataset (6400 Windows)

Used for controlled evaluation experiments.


# 4. Create a Virtual Environment (Windows)

python -m venv venv
# 5. Activate the Virtual Environment

PowerShell
venv\Scripts\Activate.ps1
Command Prompt
venv\Scripts\activate.bat


```powershell
cd path\to\CellMob
```

# 6 . Install Dependencies

First upgrade pip:

python -m pip install --upgrade pip

Then install dependencies:

pip install -r requirements.txt

# 7. What You Should Do Next

Go to the code folder first, then run the scripts in this order.

---

## 1) Data extraction / preprocessing

Run these first:

```bash
python extracting_data1.py
python 6400_KAUST.py
python standardize_and_split.py
```

- extracting_data1.py → initial data extraction / preparation  
- 6400_KAUST.py → prepares the 6400 KAUST version of the data  
- standardize_and_split.py → standardizes the features and creates the train/test split for KAUST data  

---

## 2) Model experiments

### A. 20% setting (80/20 split)

```bash
python RNN_cross_entropy_KAUST.py
python RNN_kaust_soap_ovr.py
python RNN_soap_updated.py
python RNN_soap_updated2.py
```

- RNN_cross_entropy_KAUST.py → baseline RNN with cross-entropy  
- RNN_kaust_soap_ovr.py → SOAP / AUPRC-oriented one-vs-rest experiment  
- RNN_soap_updated.py → updated SOAP variant  
- RNN_soap_updated2.py → second updated SOAP variant  

### B. Full 6400 setting

```bash
python cross_entropy.py
python soap.py
```

- cross_entropy.py → baseline 3-class cross-entropy experiment on the 6400 setup  
- soap.py → 3-class SOAP / AUPRC-oriented experiment on the 6400 setup  

---

## 3) Binary comparison experiments

Run these **after the main 3-class experiments**.

### A. Walk vs Bus

Run:

```bash
cross_entropy_approach.ipynb
SOAP_approach.py
```

- cross_entropy_approach.ipynb → binary baseline notebook for walk vs bus  
- SOAP_approach.py → binary SOAP experiment for walk vs bus  

### B. Walk vs Car

```bash
python cross-entropy.py
python soap_versionm.py
```

- cross-entropy.py → binary baseline for walk vs car  
- soap_versionm.py → binary SOAP version for walk vs car  


8. Centralized Model

In addition to the KAUST-only experiments, a centralized multi-city model was also implemented.

The centralized model combines training data from multiple cities into a single dataset. 
This model has access to all training windows and therefore represents the best-case 
performance scenario when privacy constraints are ignored.

Cities used in the centralized setup:

- KAUST
- Jeddah
- Mekkah

The centralized training script is:

Centralized_GLOBAL_model.py

What this script does:

1. Loads the training datasets from each city
2. Merges them into a single combined dataset
3. Trains an RNN classifier using cross-entropy loss
4. Evaluates the model on the combined test set

Metrics reported include:

- Confusion Matrix
- Precision–Recall curves
- AP / AUPRC per class
- Macro AUPRC

The centralized model serves as the reference performance that federated learning will be compared against.



9. Local City Models

Before running federated learning, local models were trained independently on each city 
to verify that the architecture works well within each environment.

Folders:

local model KAUST/
local model Jeddah/
local model Makkah/

Each folder contains scripts that train an RNN model using only that city's data.

This step ensures:

- the model can learn the transport patterns of each city
- the preprocessing pipeline works correctly
- the model converges before federated aggregation begins



10. Federated Learning Experiments

Federated learning experiments are located inside:

fedL/

and are executed using:

FED_learning.py

In addtion to 
feddd.py in FedL/


In the federated setup, each city acts as a separate client:

- Client 1: KAUST
- Client 2: Jeddah
- Client 3: Mekkah

Each client:

1. trains a local RNN model using its own private dataset
2. does not share raw data
3. only sends model parameters (weights) to the server

The server performs Federated Averaging (FedAvg) to update the global model.



11. Final Comparison

The following approaches are compared in this project:

- Cross-entropy baseline
- SOAP / AUPRC optimization
- Centralized multi-city model
- Federated learning model

Evaluation focuses on:

- AP / AUPRC per class
- Macro AUPRC
- Confusion matrices
- Precision–Recall curves

This allows analyzing:

1. whether AUPRC-oriented optimization improves minority class detection
2. the performance gap between centralized learning and federated learning
3. the trade-off between privacy preservation and model performance
