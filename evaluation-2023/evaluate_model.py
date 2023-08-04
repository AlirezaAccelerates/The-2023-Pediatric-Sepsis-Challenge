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
    accuracy = compute_accuracy(tn, fp, fn, tp)

    score = challenge_score(label, prediction_probability)  

    return score, accuracy


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

# Compute accuracy.
def compute_accuracy(tn, fp, fn, tp):
    return (tp + tn) / (tn + fp + fn + tp)

# Compute challenge score .
def challenge_score(labels, outputs):
    # Check the data.
    assert len(labels) == len(outputs)

    # Convert the data to NumPy arrays for easier indexing.
    labels = np.asarray(labels, dtype=np.float64)
    outputs = np.asarray(outputs, dtype=np.float64)

    num_instances = len(labels)

    # Collect the unique output values as the thresholds for the positive and negative classes.
    thresholds = np.unique(outputs)
    thresholds = np.append(thresholds, thresholds[-1]+1)
    thresholds = thresholds[::-1]
    num_thresholds = len(thresholds)

    idx = np.argsort(outputs)[::-1]

    # Initialize the TPs, FPs, FNs, and TNs with no positive outputs.
    tp = np.zeros(num_thresholds)
    fp = np.zeros(num_thresholds)
    fn = np.zeros(num_thresholds)
    tn = np.zeros(num_thresholds)

    tp[0] = 0
    fp[0] = 0
    fn[0] = np.sum(labels == 1)
    tn[0] = np.sum(labels == 0)

    # Update the TPs, FPs, FNs, and TNs using the values at the previous threshold.
    k = 0
    for l in range(1, num_thresholds):
        tp[l] = tp[l-1]
        fp[l] = fp[l-1]
        fn[l] = fn[l-1]
        tn[l] = tn[l-1]

        while k < num_instances and outputs[idx[k]] >= thresholds[l]:
            if labels[idx[k]] == 1:
                tp[l] += 1
                fn[l] -= 1
            else:
                fp[l] += 1
                tn[l] -= 1
            k += 1

        # Compute the FPRs.
        fpr = np.zeros(num_thresholds)
        for l in range(num_thresholds):
            if tp[l] + fn[l] > 0:
                fpr[l] = float(fp[l]) / float(tp[l] + fn[l])
            else:
                fpr[l] = float('nan')

        # Find the threshold such that FPR <= 0.05.
        max_fpr = 0.05
        if np.any(fpr <= max_fpr):
            l = max(l for l, x in enumerate(fpr) if x <= max_fpr)
            tp = tp[l]
            fp = fp[l]
            fn = fn[l]
            tn = tn[l]
        else:
            tp = tp[0]
            fp = fp[0]
            fn = fn[0]
            tn = tn[0]

    # Compute the TPR at FPR <= 0.05.
    if tp + fn > 0:
        max_tpr = tp / (tp + fn)
    else:
        max_tpr = float('nan')

    return max_tpr


if __name__ == "__main__":

    # Compute the scores for the model outputs.
    scores = evaluate_model(sys.argv[1], sys.argv[2])
    
    # Unpack the scores.
    score, accuracy = scores
    
    # Construct a string with scores.
    output_string = \
        'Challenge Score: {:.3f}\n'.format(score) + \
        'Accuracy: {:.3f}\n'.format(accuracy)

    
    # Output the scores to screen and/or a file.
    if len(sys.argv) == 3:
        print(output_string)
    elif len(sys.argv) == 4:
        with open(sys.argv[3], 'w') as f:
            f.write(output_string)

