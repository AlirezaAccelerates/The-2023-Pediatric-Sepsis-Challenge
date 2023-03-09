#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, os, sys
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import pickle

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

def train_challenge_model(data_folder, model_folder, verbose):
    '''
    Train model
    '''
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    features = list()
    outcomes = list()

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        # Load data.
        patient_id = patient_ids[i]
        patient_data = load_challenge_data(data_folder, patient_id)

        # Extract features.
        current_features = get_features(patient_data)
        features.append(current_features)

        # Extract labels.
        current_outcome = get_outcome(patient_data)
        outcomes.append(current_outcome)

    features = np.vstack(features)
    outcomes = np.vstack(outcomes)

    # Train the models.
    if verbose >= 1:
        print('Training the Challenge models on the Challenge data...')

    # Define parameters for random forest classifier and regressor.
    n_estimators   = 123  # Number of trees in the forest.
    max_leaf_nodes = 456  # Maximum number of leaf nodes in each tree.
    random_state   = 789  # Random state; set for reproducibility.

    # Impute any missing features; use the mean value by default
    imputer = SimpleImputer().fit(features)
    features = imputer.fit_transform(features)

    # Train the models
    model = RandomForestClassifier(n_estimators=n_estimators,
                                   max_leaf_nodes=max_leaf_nodes,
                                   random_state=random_state
                                   )
    model.fit(features, outcomes.ravel())

    # Save the models.
    save_challenge_model(model_folder, model)

    if verbose >= 1:
        print('Done.')


def load_challenge_model(model_folder, verbose):
    '''
    Load your trained models. This function is *required*. You should edit this
    function to add your code, but do *not* change the arguments of this function.
    '''
    with open(os.path.join(model_folder, 'model.pkl'),) as filename:
        model = pickle.load(filename)

    return model


def run_challenge_model(model, imputer, data_folder, patient_id, verbose):
    '''
    Run your trained models. This function is *required*. You should edit this
    function to add your code, but do *not* change the arguments of this function.
    '''
    # Load data
    patient_data = load_challenge_data(data_folder, patient_id)

    # Extract features
    features = get_features(patient_data)
    features = features.reshape(1, -1)

    # Impute missing data
    features = imputer.transform(features)

    # Apply models to features
    outcome = model.predict(features)[0]
    outcome_probability = model.predict_proba(features)[0, 1]

    return outcome, outcome_probability

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

def save_challenge_model(model_folder, model):
    '''
    Save trained challenge model
    '''
    with open(os.path.join(model_folder, 'model.pkl'), 'wb') as filename:
        pickle.dump(model, filename)


def get_features(data):
    '''
    Extract features from the data.
    '''
    # TODO: get relevant features
    return data