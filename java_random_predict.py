#!/usr/bin/env python
import random
import os
import signal
import subprocess
import time

from evaluator import aicrowd_helpers
from evaluator.learning_to_smell import L2SPredictor
from py4j.java_gateway import JavaGateway

class JavaRandomPredictor(L2SPredictor):

    """
    Below paths will be preloaded for you, you can read them as you like.
    """
    training_data_path = None
    test_data_path = None
    vocabulary_path = None

    """
    You can do any preprocessing required for your codebase here like loading up models into memory, etc.
    """
    def predict_setup(self):
        args = ['javac', '-cp', 'py4j0.10.9.1.jar', 'RandomPredictor.java']
        subprocess.call(args, cwd='assets/JavaCodebase/')
        args = ['java', '-cp', '.:py4j0.10.9.1.jar', 'RandomPredictor']
        childProc = subprocess.Popen(args, cwd='assets/JavaCodebase/')
        time.sleep(1)
        self.gateway = JavaGateway()
        self.random_predictor = self.gateway.entry_point        # get the AdditionApplication instance
        self.random_predictor.readVocabulary(self.vocabulary_path)

    """
    This function will be called for all the smiles string one by one during the evaluation.
    The return need to be a list of list. Example: [["burnt", "oily", "rose"], ["fresh", "citrus", "green"]]
    NOTE: In case you want to load your model, please do so in `predict_setup` function.
    """
    def predict(self, smile_string):
        data = self.random_predictor.predict(smile_string)
        return data

    """
    (optional)
    Add your training code in this function, this will be called only during training locally.
    This will not be executed during the evaluation, remember to save your models/pkl/artifacts properly.
    You can upload them via git (git-lfs in case of large files)
    """
    def train(self):
        raise NotImplementedError()


if __name__ == "__main__":
    submission = JavaRandomPredictor()
    submission.run()
    print("Successfully generated predictions!")
