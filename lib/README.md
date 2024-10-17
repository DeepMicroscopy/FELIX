# Source code 

## Overview

This folder contains the source code used to train a cat classifier to support the behavioral experiments with machine learning predictions.

## Contents

- `dataset.py`: Contains the dataset classes to train and evaluate the cat classifier.
- `datamodule.py`- Contains the pytorch-lightning datamodule classes for training the cat classifier.
- `model.py`: Contains the code to create a torchvision classifier pretrained with ImageNet weights. 
- `loss.py`: Contains code to train a model using LogitNormLoss. 