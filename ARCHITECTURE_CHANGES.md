# Architecture changes

The four projects keep the same task goals and Streamlit entry points, but their internal ML architectures have been adjusted so they are not identical to the original implementations.

## 1. Credit Scoring Model
- Replaced the original Logistic/Decision Tree/Random Forest comparison with Logistic Regression, Random Forest, Extra Trees, and a soft-voting hybrid ensemble.
- The saved model is still selected by test ROC-AUC, so `app.py` remains compatible.

## 2. Emotion Recognition from Speech
- Replaced the CNN + BiLSTM network with a CNN + bidirectional GRU architecture.
- Added learned temporal attention pooling instead of simple mean pooling.
- Retained MFCC input extraction and the same training/prediction workflow.

## 3. Handwritten Character Recognition
- Reworked the CNN into three stages with double-convolution blocks and adaptive average pooling.
- Reduced dependence on a fixed feature-map size while keeping MNIST/EMNIST support unchanged.

## 4. Disease Prediction
- Removed XGBoost and its native `libomp` dependency.
- Added Extra Trees plus a soft-voting ensemble combining Logistic Regression, RBF SVM, and Extra Trees.
- This is more portable on macOS while preserving the same Breast Cancer Wisconsin task and Streamlit UI.
