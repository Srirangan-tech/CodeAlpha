# CodeAlpha Task 3 — Handwritten Character Recognition

## Objective
Identify handwritten digits or alphabet characters using image processing and deep learning.

## Approach
- MNIST digits or EMNIST Letters dataset
- Multi-stage CNN with double-convolution blocks, dropout, adaptive average pooling and a dense classification head
- Validation/test accuracy tracking
- Confusion matrix
- Streamlit image-upload demo

## Train on MNIST
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --dataset mnist --epochs 8
streamlit run app.py
```

## Train on EMNIST letters
```bash
python train.py --dataset emnist_letters --epochs 10
```

The dataset is downloaded automatically by `torchvision` into `data/`.
