import numpy as np
from evaluate_model import evaluate_model


def main():
    labels = np.zeros(100)
    labels[::4] = 1
    outputs = np.zeros(100)
    outputs[::8] = 1

    challenge_score = evaluate_model(labels, outputs)


if __name__ == "__main__":
    main()