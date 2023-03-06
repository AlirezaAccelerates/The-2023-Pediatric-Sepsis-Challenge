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
    if column_names[-1] == 'inhospital_mortality':
        column_names = column_names[:-1]
        label = column_names[-1]
        data = data[:, :-1]

    return data, label
  
  
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
