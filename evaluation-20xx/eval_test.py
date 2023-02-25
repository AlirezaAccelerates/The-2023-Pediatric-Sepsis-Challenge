import numpy as np
from evaluate_model import evaluate_model


def main():
    true = np.zeros(100)
    true[::4] = 1
    pred = np.zeros(100)
    pred[::8] = 1

    challenge_score = evaluate_model(true, pred)


if __name__ == "__main__":
    main()