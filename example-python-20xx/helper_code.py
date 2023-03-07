#!/usr/bin/env python

# Do *not* edit this script.
# These are helper functions that you can use with your code.
# Check the example code to see how to import these functions to your code.

import os, numpy as np, scipy as sp, scipy.io

### Challenge data I/O functions

def load_challenge_data(data_folder):
  
    with open(data_folder, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.genfromtxt(f, delimiter='|', missing_values='NA')
        
    # Ignore inhospital_mortality column if present.
    if column_names[-1] == 'inhospital_mortality' and column_names[0] ==:
        features = column_names[1:-1]
        label = data[:,-1]
        patient_ids = data[:,0]
        data = data[:, 1:-1]
        

    return patient_ids, data, label, features
  
 
  # Save the Challenge outputs for one file.
def save_challenge_outputs(data_folder, prediction_binary, prediction_probability):
    
    # Sanitize values, e.g., in case they are a singleton array.
    prediction_binary = sanitize_boolean_value(prediction_binary)
    prediction_probability = sanitize_scalar_value(prediction_probability)
    
    if data_folder is not None:
      with open(data_folder, 'w') as f:
          f.write('PatientID|PredictedProbability|PredictedBinary\n')
          for (i, p, b) in zip(patient_ids, prediction_probability, prediction_binary):
              f.write('%d|%g|%d\n' % (i, p, b))
  

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
