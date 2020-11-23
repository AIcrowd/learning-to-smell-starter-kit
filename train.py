"""
Import implementations you want to play around with here
"""
from random_predict import RandomPredictor
from fingerprint_predict import FingerprintPredictor
from java_random_predict import JavaRandomPredictor

"""
The implementation you want to submit as your submission
"""
submission = FingerprintPredictor() # This is a class derived from the L2SPredictor class from evaluator.learning_to_smell
# submission = RandomPredictor() # This is a class derived from the L2SPredictor class from evaluator.learning_to_smell

submission.train()


print("Successfully run training for %s" % submission.__class__.__name__)
