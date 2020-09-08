#!/usr/bin/env python

import pandas as pd
import json

import os
import random


TRAINING_DATASET_PATH = os.getenv("TRAINING_DATASET_PATH", "data/train.csv")
TEST_DATASET_PATH = os.getenv("TEST_DATASET_PATH", "data/test.csv")

VOCABULARY_PATH = os.getenv("VOCABULARY_PATH", "data/vocabulary.txt")


vocabulary = open(VOCABULARY_PATH).read().split()


test_df = pd.read_csv(TEST_DATASET_PATH)


def generate_sample_submission(df, vocabulary):
    """
    Generates a sample submission file
    by randomly generating 5 random smell sentences 
    of upto 3 random smell words.

    The 5 sentences are to be separated by a ;
    and all the individual smells words in 
    a sentence are to be separated by a ,
    """

    def generate_random_prediction():
        # Generates a single prediction string

        SENTENCES = []
        for _ in range(5):
            WORDS = []
            for _ in range(3):
                WORDS.append(
                    random.choice(vocabulary)
                )
            SENTENCES.append(",".join(sorted(set(WORDS))))
        return ";".join(SENTENCES)
    
    predictions = [generate_random_prediction() for _ in range(df.shape[0])]

    return pd.DataFrame({
        "SMILES" : df.SMILES.tolist(),
        "PREDICTIONS" : predictions
    })

sample_submission_df = generate_sample_submission(test_df, vocabulary)

submission_file_path = "submission.csv"
print("Writing Sample Submission to : ", submission_file_path)
sample_submission_df.to_csv(
    submission_file_path,
    index=False
)
