#!/usr/bin/env python

# Do *not* edit this script. Changes will be discarded so that we can run the trained models consistently.

# This file contains functions for running models for the Challenge. You can run it as follows:
#
#   python run_model.py models data outputs
#
# where 'models' is a folder containing the your trained models, 'data' is a folder containing the Challenge data, and 'outputs' is a
# folder for saving your models' outputs.

import numpy as np, scipy as sp, os, sys
from helper_code import *
from team_code import load_challenge_models, run_challenge_models

# Run model.
def run_model(model_folder, data_folder, output_folder, allow_failures, verbose):
    # Load model(s).
    if verbose >= 1:
        print('Loading the Challenge models...')

    # You can use this function to perform tasks, such as loading your models, that you only need to perform once.
    models = load_challenge_models(model_folder, verbose) ### Teams: Implement this function!!!
    
    
    # Find the Challenge data.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')
        
    data, label = load_challenge_data(data_folder, model_folder)
    num_patients = len(data)

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')
        
    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
