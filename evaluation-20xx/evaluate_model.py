#!/usr/bin/env python

# Do *not* edit this script. Changes will be discarded so that we can process the models consistently.

# This file contains functions for evaluating models for the Challenge. You can run it as follows:
#
#   python evaluate_model.py labels outputs scores.csv
#
# where 'labels' is a folder containing files with the labels, 'outputs' is a folder containing files with the outputs from your
# model, and 'scores.csv' (optional) is a collection of scores for the model outputs.
#
# Each label or output file must have the format described on the Challenge webpage. The scores for the algorithm outputs are also
# described on the Challenge webpage.

import os, os.path, sys, numpy as np
import argparse
from helper_code import *


# Evaluate the models.
def evaluate_model(label_folder, output_folder):
    # Load labels and model outputs.
    _, _, label, _ = load_challenge_data(label_folder)
    patient_ids, prediction_probability, prediction_binary = load_challenge_predictions(output_folder)
    
    # Evaluate the models.
    tn, fp, fn, tp = compute_confusion_matrix(label, prediction_binary)
    sensitivity = compute_sensitivity(tp, fp)
    ppv = compute_ppv(tp, fn)
    accuracy = compute_accuracy(tn, fp, fn, tp)

    score = challenge_score(sensitivity, ppv)

    return score, sensitivity, ppv, accuracy


# Compute confusion matrix for model's predictions for the binary case
def compute_confusion_matrix(labels, predictions):

    labels = np.array(labels).astype(int)
    predictions = np.array(predictions).astype(int)

    cm = np.zeros((2, 2))
    
    for i in range(len(labels)):
        cm[labels[i]][predictions[i]] += 1

    # True Negatives
    tn = cm[0][0]
    # False Positives
    fp = cm[0][1]
    # False Negatives
    fn = cm[1][0]
    # True Positives
    tp = cm[1][1]

    return tn, fp, fn, tp


# Compute positive predictive value (PPV).
def compute_ppv(tp, fp):
    return tp / (tp + fp)

# Compute sensitivity. 
def compute_sensitivity(tp, fn):
    return tp / (tp + fn)

# Compute challenge score .
def challenge_score(sensitivity, ppv):
    return min(sensitivity, ppv)

# Compute accuracy.
def compute_accuracy(tn, fp, fn, tp):
    return (tp + tn) / (tn + fp + fn + tp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="Directory of ground truth labels")
    parser.add_argument("--outputs", required=True, help="Directory of model's outputs")
    parser.add_argument("--results", default='results', help="Directory to store model evaluation results")
    args = vars(parser.parse_args())

    os.makedirs(args['results'], exist_ok=True)
    
    # Compute the scores for the model outputs.
    score, sensitivity, ppv, accuracy = evaluate_model(args['labels'], args['outputs'])

    # Print results and write to file
    print(f"Challenge Score: {score:.3f}")
    print(f"Sensitivity: {sensitivity:.3f}")
    print(f"Positive Predictive Value: {ppv:.3f}")
    print(f"Accuracy: {accuracy:.3f}")

    with open(f"{args['results']}/results.txt", 'w') as f:
        f.write(f'Challenge Score: {score:.3f}\n')
        f.write(f'Sensitivity: {sensitivity:.3f}\n')
        f.write(f'Positive Predictive Value: {ppv:.3f}\n')
        f.write(f'Accuracy: {accuracy:.3f}\n')
