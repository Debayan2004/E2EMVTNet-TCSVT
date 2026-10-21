
# E2EMVTNet-TCSVT

Official implementation of E2EMVTNet for multi-view text spotting.

## Overview

E2EMVTNet is an end-to-end multi-view text spotting framework designed to combine information from multiple views for text detection and recognition.

The implementation is organized into modular components for feature extraction, cross-view correspondence, reliability assessment, text proposal generation, episodic memory, detection, recognition, feedback, and collaborative refinement.

## Project Structure

```text
E2EMVTNet-TCSVT/
├── configs/
│   └── config.py
├── data/
│   ├── preprocessing.py
│   ├── dataset.py
│   └── targets.py
├── models/
│   ├── mcre.py
│   ├── lecv.py
│   ├── rmw.py
│   ├── dual_branch.py
│   ├── memory.py
│   ├── recognition.py
│   ├── feedback.py
│   └── e2emvtnet.py
├── losses/
│   ├── shared.py
│   ├── detection.py
│   ├── recognition.py
│   ├── regulator.py
│   └── loss.py
├── engine/
│   ├── trainer.py
│   ├── evaluator.py
│   └── metrics.py
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

## Requirements

Python and PyTorch are required. Install the listed dependencies using:

```bash
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your system and CUDA version if GPU acceleration is required.

## Dataset

Set `DATASET_ROOT` in `configs/config.py` to the root directory containing the multi-view scene folders.

The dataset loader must follow the annotation and directory format expected by `data/dataset.py`.

## Training

After configuring the dataset path and verifying that all modules and dataset annotations are compatible, run:

```bash
python train.py
```

The training configuration uses 1000 epochs, subject to early stopping if enabled.

## Evaluation

Set `CHECKPOINT_PATH` in `evaluate.py` to the checkpoint you want to evaluate, then run:

```bash
python evaluate.py
```

## Reproducibility

For reproducible results, use the same dataset split, preprocessing, model configuration, training procedure, and evaluation protocol as the experiments reported in the paper.

## Citation

Please add the paper's final bibliographic information here after publication.
