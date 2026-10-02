# Handwritten Digit Classification from Scratch

Multiclass logistic regression implemented from scratch (NumPy + gradient descent) on the scikit-learn *digits* dataset, then compared with scikit-learn's `LogisticRegression`.

> Project for the Mathematics for Data Science course, ENSISA, 2026-2027.

## Dataset
1,797 grayscale images of handwritten digits (0 to 9), 8×8 pixels, values from 0 to 16.
Each image is flattened into 64 input features.

## Method
| Step | Choice |
|---|---|
| Preprocessing | Train/test split, standardization of the 64 inputs (fitted on the training set) |
| Model | Multiclass logistic regression (one weight vector per class + bias) |
| Training | Gradient descent written in class (batch, mini-batch or stochastic) |
| Evaluation | Accuracy and confusion matrix on the **test set** |
| Baseline | scikit-learn `LogisticRegression`, same split and preprocessing |
| Analysis | Learning rate, number of epochs, train vs test error, learned weights, errors |

## Getting started
git clone https://github.com/<user>/digits-logreg-from-scratch.git
cd digits-logreg-from-scratch
pip install -r requirements.txt
python src/regression_logistique.py

## Roadmap
- [ ] Data loading, train/test split, standardization
- [ ] Gradient descent class (from the course)
- [ ] Logistic regression from scratch
- [ ] scikit-learn baseline and comparison
- [ ] Analysis: learning rate, epochs, overfitting, learned weights, errors
- [ ] Optional: regularization (ridge / LASSO)
- [ ] Poster (deadline: 4 December 2026)

## Authors
Daris Mbau · THENLOT Yan · CLAUDE Loic
