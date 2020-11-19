import traceback
import pandas as pd
import os

from evaluator import aicrowd_helpers


class L2SPredictor:
    def __init__(self):
        self.training_data_path = os.getenv("TRAINING_DATASET_PATH", "data/train.csv")
        self.test_data_path = os.getenv("TEST_DATASET_PATH", "data/test.csv")
        self.vocabulary_path = os.getenv("VOCABULARY_PATH", "data/vocabulary.txt")
        self.predictions_output_path = os.getenv("PREDICTIONS_OUTPUT_PATH", "data/submission.csv")

    def evaluation(self):
        aicrowd_helpers.execution_start()
        self.predict_setup()

        aicrowd_helpers.execution_running()
        test_df = pd.read_csv(self.test_data_path)

        predictions = []
        for _, row in test_df.iterrows():
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
