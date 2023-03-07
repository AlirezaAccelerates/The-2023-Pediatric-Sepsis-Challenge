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
import mne
from sklearn.impute import SimpleImputer                      #??
from sklearn.ensemble import RandomForestClassifier           #??
import joblib

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find the Challenge data.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')
        
    data, label = load_challenge_data(data_folder, model_folder)
    num_patients = len(data)

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')
        
    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    
    
    # Train the models.
    if verbose >= 1:
        print('Training the Challenge models on the Challenge data...')

###########################################
                Hari
    ##############################

    # Save the models.
    save_challenge_model(model_folder, imputer, prediction_model)

    if verbose >= 1:
        print('Done!')
        
# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    print('Loading the model...')
    filename = os.path.join(model_folder, 'models.sav')
    return joblib.load(filename)

def run_challenge_models(model, data_folder, patient_id, verbose):
    imputer = model['imputer']
    prediction_model = model['prediction_model']

    # Load data.
    data, label = load_challenge_data(data_folder)
    
    # Impute missing data.
    features = imputer.transform(data)

    # Apply models to features.
    prediction_binary = prediction_model.predict(features)[0]
    prediction_probability = prediction_model.predict_proba(data)[:, 1]

    return prediction_binary, prediction_probability


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer, prediction_model):
    #####################################
    Hari
    ##################################
