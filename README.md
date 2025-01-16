# MADSP:a deep learning model based on multi-source information fusion and attention mechanism for drug combination synergy prediction

![GraphPath](https://github.com/Hhyqi/MADSP/blob/master/MADSP.png)

## Introduction

We propose MADSP, an anti-cancer drug synergy prediction method that introduces target and pathway information to provide a more comprehensive background on systems biology. First, MADSP gathers the chemical structure, target, and pathway features of drugs and employs a multi-head self-attention mechanism to learn the combined feature representation of drugs. Then, MADSP integrates the protein-protein interaction matrix with the omics data of cell lines and extracts a low-dimensional dense em-bedding of cell lines through an autoencoder. Finally, the synergy scores of drug combinations are predicted based on the embeddings of drugs and cell lines. The experiments conducted on benchmark datasets indicate that MADSP exhibits model performance superior to state-of-the-art methods. Ablation studies show that the application of multi-source information fusion and attention mechanisms significantly improves the performance of MADSP in drug synergy prediction. The case study confirms that our model can be used as a powerful tool for drug synergy prediction.

## Getting Started

### 1. Clone the repo

```
git clone https://github.com/Hhyqi/MADSP.git
```

### 2. Create conda environment

```
conda env create --name MADSP --file=environment.yml
```

## Usage

### 1. Activate the created conda environment

```
source activate MADSP
```

### 2. Train the model

```
python main.py
```

