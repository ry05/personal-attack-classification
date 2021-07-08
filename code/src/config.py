"""
Configuration file
"""

import os

# raw data
TRAIN_RAW = os.path.join("../data", "train.csv")
TEST_RAW = os.path.join("../data", "test.csv")
TEXT = 'text'

TRAIN_NUMERIC = os.path.join("../data/processed", "train_numeric_final.csv")
TEST_NUMERIC = os.path.join("../data/processed", "test_numeric_final.csv")