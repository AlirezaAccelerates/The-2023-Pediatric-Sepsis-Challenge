import argparse
from helper_code import load_challenge_labels, load_challenge_outputs
import numpy as np
import os


def compute_confusion_matrix(labels, outputs):
    '''
    Compute confusion matrix for model's predictions
    NOTE: This for the binary case
    '''
    labels = np.array(labels).astype(int)
    outputs = np.array(outputs).astype(int)

    cm = np.zeros((2, 2))
    
    for i in range(len(labels)):
        cm[labels[i]][outputs[i]] += 1

    # True Negatives
    tn = cm[0][0]
    # False Positives
    fp = cm[0][1]
    # False Negatives
    fn = cm[1][0]
    # True Positives
    tp = cm[1][1]

    return tn, fp, fn, tp


def compute_ppv(tp, fp):
    '''
    Compute positive predictive value (PPV) for model's predictions
    '''
    return tp / (tp + fp)


def compute_sensitivity(tp, fn):
    '''
    Compute the sensitivity of the model's predictions
    '''
    return tp / (tp + fn)


def evaluate_model(labels_folder, outputs_folder):
    '''
    Evaluate model performance based on metricx
    '''
    # Load labels and model outputs
    labels = load_challenge_labels(labels_folder)
    outputs = load_challenge_outputs(outputs_folder)

    tn, fp, fn, tp = compute_confusion_matrix(labels, outputs)
    sensitivity = compute_sensitivity(tp, fp)
    ppv = compute_ppv(tp, fn)

    challenge_score = min(sensitivity, ppv)

    return challenge_score, sensitivity, ppv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="Directory of ground truth labels")
    parser.add_argument("--outputs", required=True, help="Directory of model's outputs")
    parser.add_argument("--results", default='results', help="Directory to store model evaluation results")
    args = vars(parser.parse_args())

    os.makedirs(args['results'], exist_ok=True)
    
    # Compute the scores for the model outputs.
    challenge_score, sensitivity, ppv = evaluate_model(args['labels'], args['outputs'])

    # Print results and write to file
    print(f"Challenge Score: {challenge_score:.3f}")
    print(f"Sensitivity: {sensitivity:.3f}")
    print(f"Positive Predictive Value: {ppv:.3f}")

    with open(f"{args['results']}/results.txt", 'w') as f:
        f.write(f'Challenge Score: {challenge_score:.3f}\n')
        f.write(f'Sensitivity: {sensitivity:.3f}\n')
        f.write(f'Positive Predictive Value: {ppv:.3f}\n')