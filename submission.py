"""
Import implementations you want to play around with here
"""
from random_predict import RandomPredictor
from fingerprint_predict import FingerprintPredictor

"""
The implementation you want to submit as your submission
"""
submission = FingerprintPredictor()
#submission = RandomPredictor()


import __main__ as main
from evaluator.aicrowd_helpers import yesno, is_grading
if not is_grading:
    # This will not run during online evaluation
    run_train = yesno("Do you want to run training phase?")
    if run_train:
        submission.train()

submission.run()
print("Successfully generated prediction file at %s" % submission.predictions_output_path)
