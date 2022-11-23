# First Principles Softmax Regression for MNIST Classifiction

## Overview
Softmax regression is a generalization of logistic regression to the multiclass case. This project contains a softmax regression model implemented using only the NumPy library and applied to the MNIST dataset of handwritten digits. A final accuracy of 92.09% was obtained on the test set.

## Approach

The MNIST handwritten digit dataset contains 60 000 training samples and 10 000 testing samples where each sample is a 28x28 pixel image of a single handwritten digit in the range 0 to 9. The softmax regression model itself was implemented in a modular manner to faciliate simple alteration of model parameters to assist in achieving optimal performance. Several different model variations were trained and compared using the performance on the test set.

## Repository Structure

- *trainig.ipynb* - Training and evaluation of softmax regression model on MNIST dataset.
- *utils.py* - Miscellaneous output formatting functions.
- *data/* - Contains the MNIST training and test data files.