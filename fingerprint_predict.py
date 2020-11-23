#!/usr/bin/env python
"""
This implementation is port of one of community contribution done for Learning to Smell competition.

Original Post: https://discourse.aicrowd.com/t/explained-by-the-community-200-chf-cash-prize-x-5/3674/8?u=shivam
Original Author: @lacemaker 

Related Links
Medium Blog: https://medium.com/@latticetower/aicrowd-learning-to-smell-challenge-right-fingerprint-is-all-you-need-4d45e2afb869?source=friends_link&sk=74aa8b448f2d5d19e31ee32901151e37
Github Repository: https://github.com/latticetower/learning-to-smell-baseline

Don't forget to hit like and star original repository if you like this!! O:)
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

from evaluator import aicrowd_helpers
from evaluator.learning_to_smell import L2SPredictor

class FingerprintPredictor(L2SPredictor):

    """
    Below paths will be preloaded for you, you can read them as you like.
    """
    training_data_path = None
    test_data_path = None
    vocabulary_path = None

    def __init__(self):
        super().__init__()
        self.pkl_location = "assets/fingerprint_predict_knn_v1.pkl"


    def default_prediction(self):
        self.train.SENTENCE.value_counts()[:5]
        return [[i] for i in self.train.SENTENCE.value_counts()[:5].index]

    """
    You can do any preprocessing required for your codebase here like loading up models into memory, etc.
    """
    def predict_setup(self):
        self.common_setup()
        self.nbrs = pickle.load(open(self.pkl_location, 'rb'))
        pass

    """
    This function will be called for all the smiles string one by one during the evaluation.
    The return need to be a list of list. Example: [["burnt", "oily", "rose"], ["fresh", "citrus", "green"]]
    NOTE: In case you want to load your model, please do so in `predict_setup` function.
    """
    def predict(self, smile_string):
        try:
            fingerprint_bits = self.to_bits(self.fingerprints.loc[self.fingerprints['SMILES'] == smile_string].iloc[0]['fingerprint'])
        except Exception as e:
            print(e)
            fingerprint_bits = None

        if fingerprint_bits is not None:
            distances, neighbour_indices = self.nbrs.kneighbors([fingerprint_bits])
            SENTANCE = []
            for neighbour in neighbour_indices:
                for x in neighbour:
                    WORDS = self.train.loc[self.train_df.index[x], "SENTENCE"].split(',')
                    SENTANCE.append(WORDS)
            return SENTANCE
        print("Reverting to default prediction")
        return self.default_prediction()


    def common_setup(self):
        self.test = pd.read_csv(self.test_data_path)
        self.train = pd.read_csv(self.training_data_path)
        self.vocab = pd.read_csv(self.vocabulary_path, header=None)
        self.fingerprints = pd.read_csv("assets/pubchem_fingerprints.csv")

        self.train_df = self.train.merge(self.fingerprints, on="SMILES", how="left")
        self.train_df = self.train_df[~self.train_df.fingerprint.isnull()]
        print(self.train_df.fingerprint.isnull().sum(), "train molecules have no associated fingerprint")

        self.test_df = self.test.merge(self.fingerprints, on="SMILES", how="left")
        self.test_df = self.test_df[~self.test_df.fingerprint.isnull()]
        print(self.test_df.fingerprint.isnull().sum(), "test molecules have no associated fingerprint")
        pass

    """
    (optional)
    Add your training code in this function, this will be called only during training locally.
    and NOT executed during the evaluation, remember to save your models/pkl/artifacts properly.
    You can upload them via git (git-lfs in case of large files)
    """
    def train(self):
        self.common_setup()
        train_fingerprints = self.train_df.fingerprint.apply(self.to_bits)#lambda fingerprint_string: [x=='1' for x in fingerprint_string])
        train_fingerprints = np.stack(train_fingerprints.values)
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(train_fingerprints)
        nbrsPickle = open(self.pkl_location, 'wb')
        pickle.dump(nbrs, nbrsPickle)


    def to_bits(self, x):
        try:
            unpacked = np.unpackbits(np.frombuffer(bytes.fromhex(x), dtype=np.uint8))
        except Exception as e:
            print(e)
            print(x)
        return unpacked

if __name__ == "__main__":
    submission = FingerprintPredictor()
    submission.run()
    print("Successfully generated predictions!")
