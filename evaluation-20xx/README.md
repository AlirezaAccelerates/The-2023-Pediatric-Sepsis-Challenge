# Scoring code for the George B. Moody PhysioNet Challenge 2023

This repository contains the Python and MATLAB evaluation code for the George B. Moody PhysioNet Challenge 2023.

The `evaluate_model` script evaluates the outputs of your models using the evaluation metric that is described on the [webpage](https://physionetchallenges.org/20x/) for the 20xx Challenge. This script reports multiple evaluation metrics, so check the [scoring section](https://physionetchallenges.org/20xx/#scoring) of the webpage to see how we evaluate and rank your models.

## Python

You can run the Python evaluation code by installing the NumPy package and running the following command in your terminal:

    python evaluate_model.py labels outputs scores.csv

where

- `labels` (input; required) is a folder with labels for the data, such as the [training data](https://physionetchallenges.org/20xx/#data) on the PhysioNet webpage;
- `outputs` (input; required) is a folder containing files with your model's outputs for the data; and
- `scores.csv` (output; optional) is a collection of scores for your model.

## MATLAB

You can run the MATLAB evaluation code by installing Python and the NumPy package and running the following command in MATLAB:

    evaluate_model('labels', 'outputs', 'scores.csv')

where

- `labels` (input; required) is a folder containing files with the labels for the data, such as the [training data](https://physionetchallenges.org/20xx/#data) on the PhysioNet webpage;
- `outputs` (input; required) is a folder containing files with outputs produced by your model for the data; and
- `scores.csv` (output; optional) is a collection of scores for your model.

## Troubleshooting

Unable to run this code with your code? Try one of the [example codes](https://physionetchallenges.org/20xx/#submissions) on the [training data](https://physionetchallenges.org/20xx/#data). Unable to install or run Python? Try [Python](https://www.python.org/downloads/), [Anaconda](https://www.anaconda.com/products/individual), or your package manager.

## How do I learn more?

Please see the [Challenge website](https://physionetchallenges.org/20xx/) for more details. Please post questions and concerns on the [Challenge discussion forum](https://groups.google.com/forum/#!forum/physionet-challenges).

## Useful links

- [Challenge website](https://physionetchallenges.org/20xx/)
- [MATLAB example code](https://github.com/physionetchallenges/matlab-example-20xx)
- [Python example code](https://github.com/physionetchallenges/python-example-20xx)
- [Frequently asked questions (FAQ) for this year's Challenge](https://physionetchallenges.org/2023/faq/)

