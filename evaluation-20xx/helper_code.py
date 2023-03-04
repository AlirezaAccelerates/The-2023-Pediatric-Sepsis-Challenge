import csv


def load_challenge_labels(labels_folder):
    '''
    Load patient labels
    '''
    # Define file location
    data_file = f"{labels_folder}/data.csv"
    
    # Load challenge labels
    with open(data_file, 'r') as f:
        csv_reader = csv.DictReader(f)
        labels = []
        for row in csv_reader:
            label = int(row['inhospital_mortality'])
            labels.append(label)

    return labels


def load_challenge_outputs(outputs_folder):
    '''
    Load Model Predictions
    '''
    # Define file location
    outputs_file = f"{outputs_folder}/outputs.csv"
    
    # Load model outputs
    with open(outputs_file, 'r') as f:
        csv_reader = csv.DictReader(f)
        outputs = []
        for row in csv_reader:
            output = int(row['prediction'])
            outputs.append(output)

    return outputs