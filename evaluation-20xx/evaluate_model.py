import numpy as np


def compute_confusion_matrix(labels, outputs):
    '''
    Compute confusion matrix for model's predictions
    NOTE: This for the binary case
    '''
    labels = labels.astype(int)
    outputs = outputs.astype(int)

    cm = np.zeros((2, 2))
    
    for i in range(len(labels)):
        cm[labels[i]][outputs[i]] += 1

    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
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


def evaluate_model(labels, outputs):
    '''
    Evaluate model performance based on metricx
    '''
    tn, fp, fn, tp = compute_confusion_matrix(labels, outputs)
    sens = compute_sensitivity(tp, fp)
    ppv = compute_ppv(tp, fn)

    challenge_score = min(sens, ppv)

    return challenge_score