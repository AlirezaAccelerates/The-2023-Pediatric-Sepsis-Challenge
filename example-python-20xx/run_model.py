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
