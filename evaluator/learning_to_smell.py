######################################################################################
### This is read-only file so participants can run their codes locally.            ###
### It will be over-writter during the evaluation, don't make any changes to this. ###
######################################################################################

import traceback
import pandas as pd
import os
import signal
from contextlib import contextmanager

from evaluator import aicrowd_helpers


class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Prediction timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class L2SPredictor:
    def __init__(self):
        self.training_data_path = os.getenv("TRAINING_DATASET_PATH", os.getcwd() + "/data/train.csv")
        self.test_data_path = os.getenv("TEST_DATASET_PATH", os.getcwd() + "/data/test.csv")
        self.vocabulary_path = os.getenv("VOCABULARY_PATH", os.getcwd() + "/data/vocabulary.txt")
        self.predictions_output_path = os.getenv("PREDICTIONS_OUTPUT_PATH", os.getcwd() + "/data/submission.csv")
        self.predictions_setup_timeout = int(os.getenv("PREDICTION_SETUP_TIMEOUT_SECONDS", "600"))
        self.predictions_timeout = int(os.getenv("PREDICTION_TIMEOUT_SECONDS", "1"))

    def evaluation(self):
        aicrowd_helpers.execution_start()
        try:
            with time_limit(self.predictions_setup_timeout):
                self.predict_setup()
        except NotImplementedError:
            print("predict_setup doesn't exist for this run, skipping...")

        aicrowd_helpers.execution_running()
        test_df = pd.read_csv(self.test_data_path)

        predictions = []
        for _, row in test_df.iterrows():
            with time_limit(self.predictions_timeout):
                prediction_arr = self.predict(row['SMILES'])
            prediction = ';'.join([','.join(sorted(set(i))) for i in prediction_arr])
            predictions.append(prediction)

        submission_df = pd.DataFrame({
            "SMILES" : test_df.SMILES.tolist(),
            "PREDICTIONS" : predictions
        })

        submission_df.to_csv(
            self.predictions_output_path,
            index=False
        )

        aicrowd_helpers.execution_success()

    def run(self):
        try:
            self.evaluation()
        except Exception as e:
            error = traceback.format_exc()
            print(error)
            aicrowd_helpers.execution_error(error)
            if not aicrowd_helpers.is_grading():
                raise e

    """
    You can do any preprocessing required for your codebase here like loading up models into memory, etc.
    """
    def predict_setup(self):
        raise NotImplementedError

    """
    This function will be called for all the smiles string one by one during the evaluation.
    The return need to be a list of list. Example: [["burnt", "oily", "rose"], ["fresh", "citrus", "green"]]
    NOTE: In case you want to load your model, please do so in `predict_setup` function.
    """
    def predict(self, smile_string):
        raise NotImplementedError

    """
    (optional)
    Add your training code in this function, this will be called only during training locally.
    This will not be executed during the evaluation, remember to save your models/pkl/artifacts properly.
    You can upload them via git (git-lfs in case of large files)
    """
    def train(self):
        raise NotImplementedError

    """
    You can define any new helper functions in your classes
    """
