#!/usr/bin/env python
import random

from evaluator import aicrowd_helpers
from evaluator.learning_to_smell import L2SPredictor

class RandomPredictor(L2SPredictor):

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
        random.seed(42)
        self.vocabulary = open(self.vocabulary_path).read().split()
        pass

    """
    This function will be called for all the smiles string one by one during the evaluation.
    The return need to be a list of list. Example: [["burnt", "oily", "rose"], ["fresh", "citrus", "green"]]
    NOTE: In case you want to load your model, please do so in `predict_setup` function.
    """
    def predict(self, smile_string):
        SENTENCES = []
        for _ in range(5):
            WORDS = []
            for _ in range(3):
                WORDS.append(
                    random.choice(self.vocabulary)
                )
            SENTENCES.append(WORDS)
        return SENTENCES

    """
    (optional)
    Add your training code in this function, this will be called only during training locally.
    This will not be executed during the evaluation, remember to save your models/pkl/artifacts properly.
    You can upload them via git (git-lfs in case of large files)
    """
    def train(self):
        raise NotImplementedError()

if __name__ == "__main__":
    submission = RandomPredictor()
    submission.run()
    print("Successfully generated predictions!")
