#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility to generate reference embeddings for all production-use audio data
available, using the production-ready model you trained using, for example,
tools.train_prod.py. Intended for use by tools.identify.py or other
applications you might create that use your fully trained model.

Parameters
----------
data_path : string
    Relative path to the project folder containing the same dataset.txt file that you
    generated using tools.extract_csi_features.py and used to train your model.
    Example: "data/covers80"
    
model_path : string
    Relative path to the project folder containing your trained model.
    Example: "training/covers80"


Output
------
Pickle file of reference embeddings.


Created on Sat Jul 27 10:54:22 2024

@author: alan
"""

