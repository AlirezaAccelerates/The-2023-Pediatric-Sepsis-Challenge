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
