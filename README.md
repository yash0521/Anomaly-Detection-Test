# IAV Flap Anomaly Detection

A Python-based anomaly detection system for IAV flap mechanisms using a semi-supervised machine learning approach.

## Overview

This repository contains a framework for detecting anomalies in IAV flap systems. The implementation utilizes a semi-supervised learning approach to identify abnormal behavior patterns without requiring labeled anomaly examples during training.

The code generates a synthetic dataset with the `make_moons` function, which creates two interleaving half circles - one representing normal operation data and the other representing anomalies. This serves as a testbed for anomaly detection algorithms.

## Requirements

- Python3
- NumPy
- Matplotlib
- scikit-learn

## Models Implemented
This framework implements four powerful anomaly detection algorithms:

- Isolation Forest: An ensemble method that isolates observations by randomly selecting a feature and a split value
- One-Class SVM: A support vector machine that learns a boundary around normal data points
- Local Outlier Factor: A nearest neighbors method that detects samples with substantially lower density than their neighbors
- Elliptic Envelope: A method that assumes data comes from a Gaussian distribution and fits an ellipse to the central data points

Each model provides different strengths for anomaly detection, allowing for comprehensive analysis of IAV flap behavior.
