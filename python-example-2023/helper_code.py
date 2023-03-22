#!/usr/bin/env python

# Do *not* edit this script.
# These are helper functions that you can use with your code.
# Check the example code to see how to import these functions to your code.

import os, numpy as np, scipy as sp, scipy.io
import pandas as pd

### Challenge data I/O functions

def load_challenge_data(data_folder):
  
        data = pd.read_csv(data_folder)
        label = data['inhospital_mortality']
        patient_ids = data['studyid_adm']
        data = data.drop(['studyid_adm','inhospital_mortality'], axis=1)
        features = data.columns
        
        return patient_ids, data, label, features
  
 
# Save the Challenge outputs for one file.
def save_challenge_outputs(output_folder, patient_ids, prediction_binary, prediction_probability):
    
    if output_folder is not None:
      with open(output_folder, 'w') as f:
          f.write('PatientID|PredictedProbability|PredictedBinary\n')
          for (i, p, b) in zip(patient_ids, prediction_probability, prediction_binary):
              f.write('%d|%g|%d\n' % (i, p, b))
              
    return None
  

# Load the Challenge predictions for all of the patients.
def load_challenge_predictions(folder):
    with open(folder, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        predictions = np.genfromtxt(f, delimiter='|')
    
    patient_ids = predictions[:,0].astype(int)
    prediction_probability = predictions[:,1]
    prediction_binary = predictions[:,2].astype(int)
    
    return patient_ids, prediction_probability, prediction_binary
  
  
### Other helper functions

# Check if a variable is a number or represents a number.
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

# Check if a variable is an integer or represents an integer.
def is_integer(x):
    if is_number(x):
        return float(x).is_integer()
    else:
        return False

# Check if a variable is a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False

# Check if a variable is a NaN (not a number) or represents a NaN.
def is_nan(x):
    if is_number(x):
        return np.isnan(float(x))
    else:
        return False

# Remove any quotes, brackets (for singleton arrays), and/or invisible characters.
def remove_extra_characters(x):
    return str(x).replace('"', '').replace("'", "").replace('[', '').replace(']', '').replace(' ', '').strip()

# Sanitize boolean values, e.g., from the Challenge outputs.
def sanitize_boolean_value(x):
    x = remove_extra_characters(x)
    if (is_finite_number(x) and float(x)==0) or (x in ('False', 'false', 'F', 'f')):
        return 0
    elif (is_finite_number(x) and float(x)==1) or (x in ('True', 'true', 'T', 't')):
        return 1
    else:
        return float('nan')

# Santize scalar values, e.g., from the Challenge outputs.
def sanitize_scalar_value(x):
    x = remove_extra_characters(x)
    if is_number(x):
        return float(x)
    else:
        return float('nan')
